import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.nn.functional import softmax, relu
import random


def encoder(input_dimension, output_dimension):
    l1 = nn.Linear(input_dimension, output_dimension)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model


class IQN_Policy(nn.Module):
    def __init__(self,
                 self_dimension,
                 static_dimension,
                 dynamic_dimension,
                 self_feature_dimension,
                 static_feature_dimension,
                 dynamic_feature_dimension,
                 hidden_dimension,
                 action_size,
                 device='cpu',
                 seed=0):
        super().__init__()

        # Initialize dimensions
        self.self_dimension = self_dimension
        self.static_dimension = static_dimension
        self.dynamic_dimension = dynamic_dimension
        self.self_feature_dimension = self_feature_dimension
        self.static_feature_dimension = static_feature_dimension
        self.dynamic_feature_dimension = dynamic_feature_dimension
        self.concat_feature_dimension = self_feature_dimension + static_feature_dimension + dynamic_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.action_size = action_size
        self.device = device
        self.seed_id = seed

        # Set random seeds for reproducibility
        self.set_seed(seed)

        # Encoders
        self.self_encoder = encoder(self_dimension, self_feature_dimension)
        self.static_encoder = encoder(static_dimension, static_feature_dimension)
        self.dynamic_encoder = encoder(dynamic_dimension, dynamic_feature_dimension)

        # Quantile encoder
        self.K = 32  # Number of quantiles in output
        self.n = 64  # Number of cosine features
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1, 1, self.n).to(device)
        self.cos_embedding = nn.Linear(self.n, self.concat_feature_dimension)

        # Hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)

    def set_seed(self, seed):
        """
        Set random seeds for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
        """
        Calculating the cosine values depending on the number of tau samples.
        """
        taus = torch.rand(batch_size, num_tau).to(self.device).unsqueeze(-1)
        taus = taus * cvar
        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n), "cos shape is incorrect"
        return cos, taus

    def forward(self, x, num_tau=8, cvar=1.0):
        batch_size = x.shape[0]

        self_states = x[:, :self.self_dimension]
        static_states = x[:, self.self_dimension:self.self_dimension + self.static_dimension]
        dynamic_states = x[:, self.self_dimension + self.static_dimension:]

        # Encode observations as features
        self_features = self.self_encoder(self_states)
        static_features = self.static_encoder(static_states)
        dynamic_features = self.dynamic_encoder(dynamic_states)
        features = torch.cat((self_features, static_features, dynamic_features), 1)

        # Encode quantiles as features
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos = cos.view(batch_size * num_tau, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.concat_feature_dimension)

        # Pairwise product of the input feature and cosine features
        features = (features.unsqueeze(1) * cos_features).view(batch_size * num_tau, self.concat_feature_dimension)

        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        quantiles = self.output_layer(features)

        return quantiles.view(batch_size, num_tau, self.action_size), taus

    def get_constructor_parameters(self):
        return dict(self_dimension=self.self_dimension,
                    static_dimension=self.static_dimension,
                    dynamic_dimension=self.dynamic_dimension,
                    self_feature_dimension=self.self_feature_dimension,
                    static_feature_dimension=self.static_feature_dimension,
                    dynamic_feature_dimension=self.dynamic_feature_dimension,
                    hidden_dimension=self.hidden_dimension,
                    action_size=self.action_size,
                    seed=self.seed_id)

    def save(self, directory):
        # Save network parameters
        torch.save(self.state_dict(), os.path.join(directory, f"network_params.pth"))

        # Save constructor parameters
        with open(os.path.join(directory, f"constructor_params.json"), mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(), constructor_f)

        # Save RNG state
        torch.save(torch.get_rng_state(), os.path.join(directory, "rng_state.pth"))
        if torch.cuda.is_available():
            torch.save(torch.cuda.get_rng_state(), os.path.join(directory, "cuda_rng_state.pth"))

    @classmethod
    def load(cls, directory, device="cpu"):
        # Load constructor parameters
        with open(os.path.join(directory, "constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        # Create model
        model = cls(**constructor_params)

        # Load network parameters
        model_params = torch.load(os.path.join(directory, "network_params.pth"),
                                  map_location=device)
        model.load_state_dict(model_params)
        model.to(device)

        # Load RNG state
        torch.set_rng_state(torch.load(os.path.join(directory, "rng_state.pth")))
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.load(os.path.join(directory, "cuda_rng_state.pth")))

        return model