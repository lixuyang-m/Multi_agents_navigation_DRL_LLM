import torch
import torch.optim as optim
from policy.IQN_model import IQN_Policy
from policy.DQN_model import DQN_Policy
from policy.replay_buffer import ReplayBuffer
# from policy.replay_buffer import DualReplayBuffer
from marinenav_env.envs.utils.robot import Robot
import numpy as np
import random
import time
from torch.nn import functional as F 

class Agent():
    def __init__(self, 
                 self_dimension=4,
                 static_dimension=15,
                 self_feature_dimension=32,
                 static_feature_dimension=120,
                 hidden_dimension=156,
                 action_size=9,
                 dynamic_dimension=20,
                 dynamic_feature_dimension=160,
                 cooperative=True, 
                 BATCH_SIZE=32, 
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4, 
                 TAU=1.0, 
                 GAMMA=0.99,  
                 device="cuda",
                 seed=0,
                 training=True,
                 use_iqn=True,
                 use_llm=True,
                 llm_action_flag=False  # 大模型是否给出动作建
                 ):
        
        self.cooperative = cooperative
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.training = training
        self.action_size = action_size
        self.use_iqn = use_iqn
        self.use_llm = use_llm
        # self.llm_action_flag = llm_action_flag  # 大模型是否给出动作建议

        if training:
            if use_iqn:
                self.policy_local = IQN_Policy(self_dimension,
                                            static_dimension,
                                            dynamic_dimension,
                                            self_feature_dimension,
                                            static_feature_dimension,
                                            dynamic_feature_dimension,
                                            hidden_dimension,
                                            action_size,
                                            device,
                                            seed).to(device)
                self.policy_target = IQN_Policy(self_dimension,
                                            static_dimension,
                                            dynamic_dimension,
                                            self_feature_dimension,
                                            static_feature_dimension,
                                            dynamic_feature_dimension,
                                            hidden_dimension,
                                            action_size,
                                            device,
                                            seed).to(device)
            else:
                self.policy_local = DQN_Policy(self_dimension,
                                            static_dimension,
                                            dynamic_dimension,
                                            self_feature_dimension,
                                            static_feature_dimension,
                                            dynamic_feature_dimension,
                                            hidden_dimension,
                                            action_size,
                                            device,
                                            seed).to(device)
                self.policy_target = DQN_Policy(self_dimension,
                                            static_dimension,
                                            dynamic_dimension,
                                            self_feature_dimension,
                                            static_feature_dimension,
                                            dynamic_feature_dimension,
                                            hidden_dimension,
                                            action_size,
                                            device,
                                            seed).to(device)
        
            self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)

            self.memory = ReplayBuffer(BUFFER_SIZE,BATCH_SIZE)
            # self.dual_memory = DualReplayBuffer(normal_buffer_size=BUFFER_SIZE, llm_buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

    def act_dqn(self, state, eps=0.0, use_eval=True):
        state = torch.tensor([state]).float().to(self.device)
        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()
        with torch.no_grad():
            action_values = self.policy_local(state)
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action, action_values.cpu().data.numpy()

    def act(self, state, eps=0.0, cvar=1.0, use_eval=True):
        """Returns action index and quantiles 
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        # state = self.memory.state_batch([state])
        # state = self.state_to_tensor(state) 
        state = torch.tensor([state]).float().to(self.device)
        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()
        with torch.no_grad():
            quantiles, taus = self.policy_local(state, self.policy_local.K, cvar)  # 获取分位点和分位点分布
            action_values = quantiles.mean(dim=1)  # 计算动作的平均值
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        
        return action, quantiles.cpu().data.numpy(), taus.cpu().data.numpy()

    def act_adaptive(self, state, eps=0.0):
        """adptively tune the CVaR value, compute action index and quantiles
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        cvar = self.adjust_cvar(state)
        action, quantiles, taus = self.act(state, eps, cvar)
        return action, quantiles, taus, cvar

    def act_llm(self, state, eps=0.0, cvar=1.0, llm_action=None, use_eval=True):
        state = torch.tensor([state]).float().to(self.device)

        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()

        with torch.no_grad():
            quantiles, taus = self.policy_local(state, self.policy_local.K, cvar)  # 获取分位点和分位点分布
            action_values = quantiles.mean(dim=1)  # 计算动作价值期望
        self.policy_local.train()
        #     quantile_range = quantiles.max(dim=1).values - quantiles.min(dim=1).values  # 分位点范围
        #     uncertainty = quantile_range.mean().item()  # 不确定性
        
        # # 动态权重计算
        # threshold = 0.2  # 不确定性阈值
        # alpha = 0.5  # LLM最大影响权重

        # if uncertainty > threshold:
        #     llm_weight = alpha * (uncertainty - threshold) / uncertainty
        # else:
        #     llm_weight = 0.0 
        
        # print("llm_action:", action_values, llm_action, action_values.shape, llm_action.shape)

        if llm_action is not None:
            uncertainty = quantiles.std(dim=1).mean().item()
            llm_weight = 0.2
            llm_action = torch.tensor(llm_action).unsqueeze(0).to(self.device)
            llm_action_values = action_values * llm_action
        else:
            llm_action_values = None
            llm_weight = 0.0

        if llm_action is not None:
            blended_action_values = (1 - llm_weight) * action_values.cpu().data.numpy() + llm_weight * llm_action_values.cpu().data.numpy()

            blended_mean = blended_action_values.mean()
            blended_std = blended_action_values.std()
            blended_action_values = (blended_action_values - blended_mean) / blended_std

            original_mean = action_values.cpu().data.numpy().mean()
            original_std = action_values.cpu().data.numpy().std()
            blended_action_values = blended_action_values * original_std + original_mean
        else:
            blended_action_values = action_values.cpu().data.numpy()

        # if random.random() > eps:  # Exploitation phase
        #     action = np.argmax(blended_action_values)  # Select blended action
        # else:
        #     action = random.choice(np.arange(self.action_size))

        if random.random() > eps:  # Exploitation phase
            # action = np.argmax(action_values.cpu().data.numpy())  # Select blended action
            action = np.argmax(blended_action_values)
        else:
            # if llm_action is not None:
            #     # action = np.argmax(blended_action_values)

            #     # 基于大模型分布进行引导探索
            #     llm_distribution = llm_action.cpu().data.numpy().flatten()
            #     llm_distribution = llm_distribution / llm_distribution.sum()  # 确保是概率分布
            #     action = np.random.choice(np.arange(self.action_size), p=llm_distribution)
            # else:
            #     action = random.choice(np.arange(self.action_size))

            action = random.choice(np.arange(self.action_size))

        return action, quantiles.cpu().data.numpy(), taus.cpu().data.numpy(), action_values.cpu().data.numpy(), llm_action_values, llm_weight

    def adjust_cvar(self,state):
        # scale CVaR value according to the closest distance to obstacles
        
        assert len(state) == 39, "The state size does not equal 39"

        static = state[4:19]
        dynamic = state[19:] 
        
        closest_d = np.inf

        for i in range(0,len(static),3):
            if np.abs(static[i]) < 1e-3 and np.abs(static[i+1]) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(static[i:i+2]) - static[i+2] - 0.8
            closest_d = min(closest_d,dist)

        for i in range(0,len(dynamic),4):
            if np.abs(dynamic[i]) < 1e-3 and np.abs(dynamic[i+1]) < 1e-3:
                # padding
                continue
            dist = np.linalg.norm(dynamic[i:i+2]) - 1.6
            closest_d = min(closest_d,dist)
        
        cvar = 1.0
        if closest_d < 10.0:
            cvar = closest_d / 10.0

        return cvar

    def train(self, rl_values, llm_values, llm_weight):
        if self.use_iqn:
            if self.use_llm:
                return self.train_IQN(rl_values, llm_values, llm_weight)
            else:
                return self.train_IQN(rl_values, llm_values, llm_weight)
        else:
            return self.train_DQN()
    
    def train_IQN(self, action_values=None, llm_action_values=None, llm_weight=None):
        """Update value parameters using given batch of experience tuples
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # print("action_values:", action_values, "llm_action_values:", llm_action_values, "llm_weight:", llm_weight)
        # llm_sample_rate = 0.1
        # states, actions, rewards, next_states, dones = self.dual_memory.sample(llm_sample_rate)

        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        self.optimizer.zero_grad()
        # Get max predicted Q values (for next states) from target model
        Q_targets_next,_ = self.policy_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)

        # # Compute Q targets for current states
        # if llm_action_values is not None:
        #     blended_targets = (1 - llm_weight) * Q_targets_next + llm_weight * llm_action_values
        #     Q_targets = rewards.unsqueeze(-1) + (self.GAMMA * blended_targets * (1. - dones.unsqueeze(-1)))
        # else:
        #     Q_targets = rewards.unsqueeze(-1) + (self.GAMMA * Q_targets_next * (1. - dones.unsqueeze(-1)))

        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA * Q_targets_next * (1. - dones.unsqueeze(-1)))

        # Get expected Q values from local model
        Q_expected,taus = self.policy_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0

        # if self.llm_action_flag:
        #     action_probs = F.softmax(action_values, dim=-1)
        #     llm_action_probs = F.softmax(llm_action_values, dim=-1)
        #     kl_loss = F.kl_div(action_probs.log(), llm_action_probs, reduction="batchmean")
        #     original_loss = quantil_l.sum(dim=1).mean(dim=1)

        #     alpha_min = 0.1  # 大模型建议的最低权重
        #     alpha_0 = 1  # 初始完全依赖大模型权重
        #     k = 1.5e-7  # 线性衰减系数
        #     alpha = max(alpha_min, alpha_0 - k * t)
        #     loss = original_loss + alpha * kl_loss
        # else:
        #     loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        loss = loss.mean()

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()
    
    def train_DQN(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        self.optimizer.zero_grad()

        # compute target values
        Q_targets_next = self.policy_target(next_states)
        Q_targets_next,_ = Q_targets_next.max(dim=1,keepdim=True)
        Q_targets = rewards + (1-dones) * self.GAMMA * Q_targets_next

        # compute expected values
        Q_expected = self.policy_local(states)
        Q_expected = Q_expected.gather(1,actions)

        # compute huber loss
        loss = F.smooth_l1_loss(Q_expected,Q_targets)

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()



    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def save_latest_model(self,directory):
        self.policy_local.save(directory)

    def load_model(self,path,agent_type="cooperative",device="cpu"):
        if self.use_iqn:
            self.policy_local = IQN_Policy.load(path,device)
        else:
            self.policy_local = DQN_Policy.load(path,device)


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss