"""
Some system parameters
"""
import numpy as np



class Replay:
    """
    Simulator for EMs and MUs.
    """
    def __init__(self, num_VNO = 1, episode_length = 100, obs_size=4, capacity=10000) -> None:
        """
        Initialize the simulator.
        """
        self.num_VNO = num_VNO
        self.episode_length = episode_length
        self.obs_size = obs_size
        self.capacity = capacity

        self.actions_NO = np.empty((self.capacity, self.episode_length), dtype=np.uint8)
        self.rewards_NO = np.empty((self.capacity, self.episode_length), dtype=np.float32)
        self.obs_NO = np.empty((self.capacity, self.episode_length, self.obs_size), dtype=np.float32)
        self.next_obs_NO = np.empty((self.capacity, self.episode_length, self.obs_size), dtype=np.float32)

        self.count_NO = 0
        self.index_NO = 0

        self.actions_VNO = np.empty((self.capacity, self.episode_length, self.num_VNO), dtype=np.uint8)
        self.rewards_VNO = np.empty((self.capacity, self.episode_length, self.num_VNO), dtype=np.float32)
        self.obs_VNO = np.empty((self.capacity, self.episode_length, self.num_VNO, self.obs_size), dtype=np.float32)
        self.next_obs_VNO = np.empty((self.capacity, self.episode_length, self.num_VNO, self.obs_size),
                                     dtype=np.float32)

        self.count_VNO = 0
        self.index_VNO = 0

    def store(self, obs_NO, action_NO, reward_NO, next_obs_NO, obs_VNO, action_VNO, reward_VNO, next_obs_VNO):
        """ store an episode of experiences

        """
        #print(action_NO.shape)
        #print(self.episode_length)
        assert action_NO.shape == (self.episode_length, )
        assert obs_NO.shape == (self.episode_length, self.obs_size)
        assert action_VNO.shape == (self.episode_length, self.num_VNO)
        assert obs_VNO.shape == (self.episode_length, self.num_VNO, self.obs_size)

        self.actions_NO[self.index_NO] = action_NO
        self.rewards_NO[self.index_NO] = reward_NO
        self.obs_NO[self.index_NO] = obs_NO
        self.next_obs_NO[self.index_NO] = next_obs_NO

        self.count_NO = max(self.count_NO, self.index_NO + 1)
        self.index_NO = (self.index_NO + 1) % self.capacity

        self.actions_VNO[self.index_VNO] = action_VNO
        self.rewards_VNO[self.index_VNO] = reward_VNO
        self.obs_VNO[self.index_VNO] = obs_VNO
        self.next_obs_VNO[self.index_VNO] = next_obs_VNO

        self.count_VNO = max(self.count_VNO, self.index_VNO + 1)
        self.index_VNO = (self.index_VNO + 1) % self.capacity

    def sample(self, batch_size, trace_length):
        """ sample a mini-batch of experience trajectories

        Args:
            batch_size ([type]): [description]
            trace_length ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.count_NO < batch_size:
            return
        else:
            episode_indexes = np.random.choice(range(0, self.count_NO), batch_size, replace=False)
            start_time_steps = np.random.choice(range(0, self.episode_length - trace_length), batch_size)

        actions_NO = np.empty((batch_size, trace_length), dtype=np.uint8)
        rewards_NO = np.empty((batch_size, trace_length), dtype=np.float32)
        obs_NO = np.empty((batch_size, trace_length, self.obs_size), dtype=np.float32)
        next_obs_NO = np.empty((batch_size, trace_length, self.obs_size), dtype=np.float32)

        i = 0
        for episode_index, start_time_step in zip(episode_indexes, start_time_steps):
            actions_NO[i] = self.actions_NO[episode_index, start_time_step:start_time_step + trace_length]
            rewards_NO[i] = self.rewards_NO[episode_index, start_time_step:start_time_step + trace_length]
            obs_NO[i] = self.obs_NO[episode_index, start_time_step:start_time_step + trace_length, :]
            next_obs_NO[i] = self.next_obs_NO[episode_index, start_time_step:start_time_step + trace_length, :]
            i += 1

        assert i == batch_size

        if self.count_VNO < batch_size:
            return
        else:
            episode_indexes = np.random.choice(range(0, self.count_VNO), batch_size, replace=False)
            start_time_steps = np.random.choice(range(0, self.episode_length - trace_length), batch_size)

        actions_VNO = np.empty((batch_size, trace_length, self.num_VNO), dtype=np.uint8)
        rewards_VNO = np.empty((batch_size, trace_length, self.num_VNO), dtype=np.float32)
        obs_VNO = np.empty((batch_size, trace_length, self.num_VNO, self.obs_size), dtype=np.float32)
        next_obs_VNO = np.empty((batch_size, trace_length, self.num_VNO, self.obs_size), dtype=np.float32)

        i = 0
        for episode_index, start_time_step in zip(episode_indexes, start_time_steps):
            actions_VNO[i] = self.actions_VNO[episode_index, start_time_step:start_time_step + trace_length, :]
            rewards_VNO[i] = self.rewards_VNO[episode_index, start_time_step:start_time_step + trace_length, :]
            obs_VNO[i] = self.obs_VNO[episode_index, start_time_step:start_time_step + trace_length, :, :]
            next_obs_VNO[i] = self.next_obs_VNO[episode_index, start_time_step:start_time_step + trace_length, :, :]
            i += 1

        assert i == batch_size

        return obs_NO, actions_NO, rewards_NO, next_obs_NO, obs_VNO, actions_VNO, rewards_VNO, next_obs_VNO
