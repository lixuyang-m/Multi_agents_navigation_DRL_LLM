import random
from collections import deque
import torch
import copy

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self,item):
        """Add a new experience to memory."""
        self.memory.append(item)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in samples:
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

    def size(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# import random
# from collections import deque
# import torch
# import copy
#
#
# class ReplayBuffer:
#     """Replay buffer to store experience tuples."""
#
#     def __init__(self, buffer_size, batch_size):
#         """
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """
#         self.memory = deque(maxlen=buffer_size)  # Fixed-size buffer
#         self.batch_size = batch_size
#
#     def add(self, item):
#         """Add a new experience to memory."""
#         self.memory.append(item)
#
#     def sample(self, sample_size=None):
#         """
#         Randomly sample a batch of experiences from memory.
#
#         Params
#         ======
#             sample_size (int): number of samples to draw (default: self.batch_size)
#
#         Returns
#         ======
#             A tuple of (states, actions, rewards, next_states, dones)
#         """
#         if sample_size is None:
#             sample_size = self.batch_size
#
#         samples = random.sample(self.memory, k=min(sample_size, len(self.memory)))
#         states, actions, rewards, next_states, dones = [], [], [], [], []
#         for sample in samples:
#             state, action, reward, next_state, done = sample
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             next_states.append(next_state)
#             dones.append(done)
#
#         return states, actions, rewards, next_states, dones
#
#     def size(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)
#
#
# class DualReplayBuffer:
#     """
#     Dual replay buffer to manage normal experiences and LLM-guided experiences.
#     """
#
#     def __init__(self, normal_buffer_size, llm_buffer_size, batch_size):
#         """
#         Params
#         ======
#             normal_buffer_size (int): max size of the normal experience buffer
#             llm_buffer_size (int): max size of the LLM-guided experience buffer
#             batch_size (int): size of each training batch
#         """
#         self.normal_memory = ReplayBuffer(normal_buffer_size, batch_size)
#         self.llm_memory = ReplayBuffer(llm_buffer_size, batch_size)
#         self.batch_size = batch_size
#
#     def add(self, item, llm_guided=False):
#         """
#         Add an experience to the appropriate buffer.
#
#         Params
#         ======
#             item: experience to add (state, action, reward, next_state, done)
#             llm_guided (bool): whether the experience is LLM-guided
#         """
#         if llm_guided:
#             self.llm_memory.add(item)
#         else:
#             self.normal_memory.add(item)
#
#     def sample(self, llm_ratio=0.1):
#         """
#         Sample a batch with a mix of normal and LLM-guided experiences.
#
#         Params
#         ======
#             llm_ratio (float): proportion of the batch to sample from the LLM buffer (0 to 1)
#
#         Returns
#         ======
#             A tuple of (states, actions, rewards, next_states, dones)
#         """
#         # Calculate sample sizes for each buffer
#         llm_sample_size = int(self.batch_size * llm_ratio)
#         normal_sample_size = self.batch_size - llm_sample_size
#
#         actual_llm_sample_size = min(llm_sample_size, self.llm_memory.size())
#         additional_normal_sample_size = llm_sample_size - actual_llm_sample_size
#         actual_normal_sample_size = normal_sample_size + additional_normal_sample_size
#
#         # Sample from LLM memory
#         llm_states, llm_actions, llm_rewards, llm_next_states, llm_dones = self.llm_memory.sample(actual_llm_sample_size)
#
#         # Sample from normal memory
#         normal_states, normal_actions, normal_rewards, normal_next_states, normal_dones = self.normal_memory.sample(
#             actual_normal_sample_size
#         )
#
#         # Combine samples
#         states = llm_states + normal_states
#         actions = llm_actions + normal_actions
#         rewards = llm_rewards + normal_rewards
#         next_states = llm_next_states + normal_next_states
#         dones = llm_dones + normal_dones
#
#         # Shuffle the combined batch
#         combined = list(zip(states, actions, rewards, next_states, dones))
#         random.shuffle(combined)
#
#         # Unzip and return shuffled results
#         states, actions, rewards, next_states, dones = zip(*combined)
#         return states, actions, rewards, next_states, dones
#
#     def size(self):
#         """
#         Return the total size of both buffers.
#
#         Returns
#         ======
#             int: total number of experiences in both buffers
#         """
#         return self.normal_memory.size() + self.llm_memory.size()


    

