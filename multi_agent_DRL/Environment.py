import torch
import numpy as np
import cmath
"""
Some system parameters
"""




class Environment:
    """
    Simulator for EMs and MUs.
    """
    def __init__(self, num_EMs = 10, num_MUs = 1E+8, Q_0 = 30, Q_1 = 50, theta_m = 50, h=1, w=0.275,
                 A = 1E+10, F_0 = 30, numVNO = 1) -> None:
        """
        Initialize the simulator.
        """
        self.num_EMs = num_EMs
        self.num_MUs = num_MUs
        self.Q_0 = Q_0
        self.Q_1 = Q_1
        self.theta_m = theta_m
        self.h = h
        self.w = w
        self.c = 0.7
        self.F_1 = 100
        self.p = 0.9
        self.A = A
        self.F_0 = F_0
        self.obs_space = 4
        self.numVNO = numVNO
        self.obs_space = 4
        self.act_space = 16
        self.B = 10**11

        self.N_0 = 5 * 10 ** 7
        self.N_1 = 5 * 10 ** 7
        self.G = 5 * 10 ** 9
        self.B_1 = 6 * 10 ** 9
        self.m_total = 5 * 10 ** 9
        self.reward_NO = 0.6
        self.reward_VNO = 0.6
        self.reward_norm = 10 ** 10

        self.c_benchmark = 0.7
        self.F_1_benchmark = 100
        self.p_benchmark = 0.9
        self.reward_NO_benchmark = 0.6
        self.reward_VNO_benchmark = 0.6

    @property
    def observation_space(self, single=True):
        if single:
            return self.obs_space
        else:
            return (self.obs_space, self.numVNO)

    @property
    def action_space_NO(self, single=True):
        if single:
            return self.act_space
        else:
            return (self.act_space, self.numVNO)

    @property
    def action_space_VNO(self, single=True):
        if single:
            return self.act_space * self.act_space
        else:
            return (self.act_space * self.act_space, self.numVNO)

    def reset(self):
        """
        Reset (initialize) the environment and return the initial observation
        """
        self.num_EMs = 10
        self.num_MUs = 1E+8
        self.Q_0 = 30
        self.Q_1 = 50
        self.theta_m = 50
        self.h = 1
        self.w = 0.275
        self.A = 1E+10

        actions = [2.5, 120, 10]
        self.c = actions[0]
        self.F_1 = actions[1]
        self.p = actions[2]

        # we assume that theta_1 < theta_m
        observations = np.zeros((self.numVNO + 1, self.obs_space))
        if self.F_1 < 2 * self.h * self.Q_1 ** 0.5 / self.w * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5) + self.F_0:
            theta_2 = (self.F_1 - self.F_0) / (2 * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5))
            theta_1 = self.h * self.Q_1 ** 0.5 / self.w
            self.N_0 = self.num_MUs * theta_2 / self.theta_m
            self.N_1 = self.num_MUs * (1 - theta_2 / self.theta_m)
            self.G = self.num_MUs / self.theta_m * (
                        (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) - (
                            1 / 3 * theta_1 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_1 / self.w))
            self.B_1 = self.num_MUs * (theta_1 - theta_2) * self.Q_1 / self.theta_m + self.num_MUs / self.theta_m \
                       * 1 / 3 * (
                        self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_1 ** 3 * self.w ** 2 / (self.h ** 2))
            if self.A - self.p * self.G > 0:
                self.m_total = self.G * (self.A - self.p * self.G) / 2 * self.num_EMs
                if self.m_total > self.G:
                    self.m_total = self.G
            else:
                self.m_total = 5 * 10 ** 8
            observations[0] = [self.N_0, self.B_1, self.F_1, 0]
            observations[1] = [self.G, self.N_1, self.m_total, self.c]
            return observations
        else:
            theta_3 = self.h / self.w * (
                        self.Q_0 ** 0.5 + (self.Q_0 - self.Q_1 - self.w / self.h * (self.F_0 - self.F_1)) ** 0.5)
            self.N_0 = self.num_MUs * theta_3 / self.theta_m
            self.N_1 = self.num_MUs * (1 - theta_3 / self.theta_m)
            self.G = self.num_MUs / self.theta_m * (
                        (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) - (
                            1 / 3 * theta_3 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_3 / self.w))
            self.B_1 = self.num_MUs / self.theta_m * 1 / 3 * (
                        self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_3 ** 3 * self.w ** 2 / (self.h ** 2))
            if self.A - self.p * self.G > 0:
                self.m_total = self.G * (self.A - self.p * self.G) / 2 * self.num_EMs
                if self.m_total > self.G:
                    self.m_total = self.G
            else:
                self.m_total = 5 * 10 ** 8
            observations[0] = [self.N_0, self.B_1, self.F_1, 0]
            observations[1] = [self.G, self.N_1, self.m_total, self.c]
            return observations


    def observation(self):
        """
        Return the joint observation of all EMs and MUs.

        Observation of single VNO includes:
            The number of ads chosen by users
            The number of users who subscribes its data plan
            The ads purchased by each EM
            The price of NO's mobile data

        Observation of single NO includes:
            The number of users who subscribes its data plan
            The data amount purchased by VNO
            The price of VNO's data plan
        """
        # we assume that theta_1 < theta_m
        observations = np.zeros((self.numVNO+1, self.obs_space))
        if self.F_1 < 2 * self.h * self.Q_1**0.5 / self.w * (self.Q_1**0.5 - self.Q_0**0.5) + self.F_0:
            theta_2 = (self.F_1 - self.F_0) / (2 * (self.Q_1**0.5 - self.Q_0**0.5))
            theta_1 = self.h * self.Q_1**0.5 / self.w
            self.N_0 = self.num_MUs * theta_2 / self.theta_m
            self.N_1 = self.num_MUs * (1 - theta_2 / self.theta_m)
            self.G = self.num_MUs / self.theta_m * \
                     ((1/3 * self.theta_m**3 * self.w / (self.h**2) - self.Q_1 * self.theta_m / self.w) -
                     (1/3 * theta_1**3 * self.w / (self.h**2) - self.Q_1 * theta_1 / self.w))
            self.B_1 = self.num_MUs * (theta_1 - theta_2) * self.Q_1 / self.theta_m + self.num_MUs / self.theta_m * \
                1/3 * (self.theta_m**3 * self.w**2 / (self.h**2) - theta_1**3 * self.w**2 / (self.h**2))
            """
            if self.A - self.p * self.G >= 0:
                self.m_total = self.G * (self.A - self.p * self.G) / 2 * self.num_EMs
                if self.m_total > self.G:
                    self.m_total = self.G
            else:
                self.m_total = 5 * 10 ** 8
            """
            self.m_total = self.G
            observations[0] = [self.N_0, self.B_1, self.F_1, 0]
            observations[1] = [self.G, self.N_1, self.m_total, self.c]
            return observations
        else:
            theta_3 = self.h / self.w * (self.Q_0**0.5 + (self.Q_0 - self.Q_1 - self.w / self.h *
                                                          (self.F_0 - self.F_1))**0.5)
            self.N_0 = self.num_MUs * theta_3 / self.theta_m
            self.N_1 = self.num_MUs * (1 - theta_3 / self.theta_m)
            self.G = self.num_MUs / self.theta_m * (
                    (1/3 * self.theta_m**3 * self.w / (self.h**2) - self.Q_1 * self.theta_m/self.w) -
                                (1/3 * theta_3**3 * self.w / (self.h**2) - self.Q_1 * theta_3 / self.w))
            self.B_1 = self.num_MUs / self.theta_m * 1/3 * (self.theta_m**3 * self.w**2 / (self.h**2) -
                                                       theta_3**3 * self.w**2 / (self.h**2))
            """
            if self.A - self.p * self.G >= 0:
                self.m_total = self.G * (self.A - self.p * self.G) / 2 * self.num_EMs
                if self.m_total > self.G:
                    self.m_total = self.G
            else:
                self.m_total = 5 * 10 ** 8
            """
            self.m_total = self.G
            observations[0] = [self.N_0, self.B_1, self.F_1, 0]
            observations[1] = [self.G, self.N_1, self.m_total, self.c]
            return observations




    def step(self, actions, actions_bench):
        """
        Update the decisions of all MUs and EMs
        feedback next joint observation
        :param actions: actions of the NO and VNO
            action of NO:
                The price of NO's mobile data
            action of VNO:
                The price of VNO's data plan
                The price of VNO's ad
        :return:
            N_0: number of MUs who choose NO's data plan for NO
            N_1: number of MUs who choose VNO's data plan for VNO
            m_l: number of ads purchased by each EM from VNO for VNO
            x^*: number of ads decided by all MUs for VNO
        """
        #c~[0.1,1.5]
        self.c = actions[0] / self.action_space_NO * 1 + 0.2
        #F_1~[40-160]
        self.F_1 = actions[1] / self.action_space_NO * 50 / self.action_space_NO + 100
        #p~[0.1,2]
        self.p = (actions[1] % self.action_space_NO) * 2 / self.action_space_NO + 0.1

        # c~[0.5,3]
        self.c_benchmark = actions_bench[0] * 1 / self.action_space_NO + 0.5
        # F_1~[80-200]
        self.F_1_benchmark = actions_bench[1] / self.action_space_NO * 50 / self.action_space_NO + 100
        # p~[1,10]
        self.p_benchmark = (actions_bench[1] % self.action_space_NO) * 2 / self.action_space_NO + 0.1

        self.reward_NO_benchmark, self.reward_VNO_benchmark = self.calculate_reward_benchmark()

        reward = self.calculate_reward()
        joint_observation = self.observation()

        return joint_observation, reward


    def calculate_reward(self):
        """
        Reward
        """
        # we assume that theta_1 < theta_m
        if self.F_1 < 2 * self.h * self.Q_1 ** 0.5 / self.w * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5) + self.F_0:
            reward_0 = self.N_0 * self.F_0 + self.c * self.B_1
            reward_1 = self.N_1 * self.F_1 + self.p * self.m_total - self.c * self.B_1
            reward_0 = reward_0 * 1.0 / self.reward_norm
            reward_1 = reward_1 * 1.0 / self.reward_norm
            self.reward_NO = reward_0
            self.reward_VNO = reward_1
            return reward_0, reward_1
        else:
            reward_0 = self.N_0 * self.F_0 + self.c * self.B_1
            reward_1 = self.N_1 * self.F_1 + self.p * self.m_total - self.c * self.B_1
            reward_0 = reward_0 * 1.0 / self.reward_norm
            reward_1 = reward_1 * 1.0 / self.reward_norm
            self.reward_NO = reward_0
            self.reward_VNO = reward_1
            return reward_0, reward_1

    def calculate_reward_benchmark(self):
        """
        benchmark reward
        """
        # we assume that theta_1 < theta_m
        if self.F_1_benchmark < 2 * self.h * self.Q_1 ** 0.5 / self.w * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5) + self.F_0:
            theta_2 = (self.F_1_benchmark - self.F_0) / (2 * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5))
            theta_1 = self.h * self.Q_1 ** 0.5 / self.w
            N_0 = self.num_MUs * theta_2 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_2 / self.theta_m)
            G = self.num_MUs / self.theta_m * \
                     ((1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) -
                      (1 / 3 * theta_1 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_1 / self.w))
            B_1 = self.num_MUs * (theta_1 - theta_2) * self.Q_1 / self.theta_m + self.num_MUs / self.theta_m * \
                       1 / 3 * (self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_1 ** 3 * self.w ** 2 / (
                        self.h ** 2))
            """
            if self.A - self.p_benchmark * G >= 0:
                m_total = G * (self.A - self.p_benchmark * G) / 2 * self.num_EMs
                if m_total > G:
                    m_total = G
            else:
                m_total = 5 * 10 ** 8
            """
            m_total= G
            reward_0_benchmark = N_0 * self.F_0 + self.c_benchmark * B_1
            reward_1_benchmark = N_1 * self.F_1_benchmark + self.p_benchmark * m_total - self.c_benchmark * B_1
            reward_0_benchmark = reward_0_benchmark * 1.0 / self.reward_norm
            reward_1_benchmark = reward_1_benchmark * 1.0 / self.reward_norm
            self.reward_NO_benchmark = reward_0_benchmark
            self.reward_VNO_benchmark = reward_1_benchmark
            return reward_0_benchmark, reward_1_benchmark
        else:
            theta_3 = self.h / self.w * (self.Q_0 ** 0.5 + (self.Q_0 - self.Q_1 - self.w / self.h *
                                                            (self.F_0 - self.F_1_benchmark)) ** 0.5)
            N_0 = self.num_MUs * theta_3 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_3 / self.theta_m)
            G = self.num_MUs / self.theta_m * (
                    (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) -
                    (1 / 3 * theta_3 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_3 / self.w))
            B_1 = self.num_MUs / self.theta_m * 1 / 3 * (self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) -
                                                              theta_3 ** 3 * self.w ** 2 / (self.h ** 2))
            """
            if self.A - self.p_benchmark * G >= 0:
                m_total = G * (self.A - self.p_benchmark * G) / 2 * self.num_EMs
                if m_total > G:
                    m_total = G
            else:
                m_total = 5 * 10 ** 8
            """
            m_total = G
            reward_0_benchmark = N_0 * self.F_0 + self.c_benchmark * B_1
            reward_1_benchmark = N_1 * self.F_1_benchmark + self.p_benchmark * m_total - self.c_benchmark * B_1
            reward_0_benchmark = reward_0_benchmark * 1.0 / self.reward_norm
            reward_1_benchmark = reward_1_benchmark * 1.0 / self.reward_norm
            self.reward_NO_benchmark = reward_0_benchmark
            self.reward_VNO_benchmark = reward_1_benchmark
            return reward_0_benchmark, reward_1_benchmark


