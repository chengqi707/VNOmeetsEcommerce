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
    def __init__(self, num_EMs = 10, num_MUs = 1E+7, Q_0 = 30, Q_1 = 50, theta_m = 50, h=1, w=0.2,
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
        self.c = 0
        self.F_1 = 0
        self.p = 0
        self.A = A
        self.F_0 = F_0
        self.obs_space = 4
        self.numVNO = numVNO
        self.obs_space = 4
        self.act_space = 16
        self.B = 10**11

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
        self.num_MUs = 1E+7
        self.Q_0 = 30
        self.Q_1 = 50
        self.theta_m = 50
        self.h = 1
        self.w = 0.2
        self.A = 1E+10

        actions=[2.5,120,10**7]
        self.c = actions[0]
        self.F_1 = actions[1]
        self.p = actions[2]

        # we assume that theta_1 < theta_m
        observations = np.zeros((self.numVNO + 1, self.obs_space))
        if self.F_1 < 2 * self.h * self.Q_1 ** 0.5 / self.w * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5) + self.F_0:
            theta_2 = (self.F_1 - self.F_0) / (2 * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5))
            theta_1 = self.h * self.Q_1 ** 0.5 / self.w
            N_0 = self.num_MUs * theta_2 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_2)
            G = self.num_MUs * (
                        (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) - (
                            1 / 3 * theta_1 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_1 / self.w))
            B_1 = self.num_MUs * (theta_1 - theta_2) * self.Q_1 / self.theta_m + self.num_MUs / self.theta_m * 1 / 3 * (
                        self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_1 ** 3 * self.w ** 2 / (self.h ** 2))
            m_total = G * (self.A - self.p * G) / 2 * self.num_EMs
            # return N_0,B_1,self.F_1,G,N_1,m_total,self.c
            observations[0] = [N_0, B_1, self.F_1, 0]
            observations[1] = [G, N_1, m_total, self.c]
            return observations
        else:
            theta_3 = self.h / self.w * (
                        self.Q_0 ** 0.5 + (self.Q_0 - self.Q_1 - self.w / self.h * (self.F_0 - self.F_1)) ** 0.5)
            N_0 = self.num_MUs * theta_3 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_3)
            G = self.num_MUs * (
                        (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) - (
                            1 / 3 * theta_3 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_3 / self.w))
            B_1 = self.num_MUs / self.theta_m * 1 / 3 * (
                        self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_3 ** 3 * self.w ** 2 / (self.h ** 2))
            m_total = G * (self.A - self.p * G) / 2 * self.num_EMs
            # return N_0,B_1,self.F_1,G,N_1,m_total,self.c
            observations[0] = [N_0, B_1, self.F_1, 0]
            observations[1] = [G, N_1, m_total, self.c]
            return observations

    # todo normalize the observations
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
        if self.F_1<2*self.h*self.Q_1**0.5/self.w*(self.Q_1**0.5-self.Q_0**0.5)+self.F_0:
            theta_2 = (self.F_1-self.F_0)/(2*(self.Q_1**0.5-self.Q_0**0.5))
            theta_1 = self.h*self.Q_1**0.5/self.w
            N_0 = self.num_MUs * theta_2 / self.theta_m
            N_1 = self.num_MUs * (1-theta_2)
            G=self.num_MUs* ((1/3*self.theta_m**3*self.w/(self.h**2)-self.Q_1*self.theta_m/self.w)-(1/3*theta_1**3*self.w/(self.h**2)-self.Q_1*theta_1/self.w))
            B_1=self.num_MUs*(theta_1-theta_2)*self.Q_1/self.theta_m+self.num_MUs/self.theta_m*1/3*(self.theta_m**3*self.w**2/(self.h**2)-theta_1**3*self.w**2/(self.h**2))
            m_total=G*(self.A-self.p*G)/2*self.num_EMs
            #return N_0,B_1,self.F_1,G,N_1,m_total,self.c
            observations[0]=[N_0,B_1,self.F_1,0]
            observations[1]=[G,N_1,m_total,self.c]
            return observations


        else:
            theta_3=self.h/self.w*(self.Q_0**0.5 + (self.Q_0-self.Q_1-self.w/self.h*(self.F_0-self.F_1))**0.5)
            N_0 = self.num_MUs * theta_3 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_3)
            G=self.num_MUs* ((1/3*self.theta_m**3*self.w/(self.h**2)-self.Q_1*self.theta_m/self.w)-(1/3*theta_3**3*self.w/(self.h**2)-self.Q_1*theta_3/self.w))
            B_1=self.num_MUs/self.theta_m*1/3*(self.theta_m**3*self.w**2/(self.h**2)-theta_3**3*self.w**2/(self.h**2))
            m_total = G * (self.A - self.p * G) / 2 * self.num_EMs
            # return N_0,B_1,self.F_1,G,N_1,m_total,self.c
            observations[0] = [N_0, B_1, self.F_1, 0]
            observations[1] = [G, N_1, m_total, self.c]
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
        #c~[0.5,3]
        self.c = actions[0]*2.5/self.action_space_NO+0.5
        #F_1~[80-200]
        self.F_1 = actions[1]/self.action_space_NO*120/self.action_space_NO+80
        #p~[10**10/8000,10*10/1000]
        self.p = (actions[1] % self.action_space_NO)*10**7/self.action_space_NO+10**6

        reward = self.calculate_reward()
        joint_observation = self.observation()

        return joint_observation, reward


    def calculate_reward(self):
        """
        Reward
        """
        # we assume that theta_1 < theta_m
        if self.F_1 < 2 * self.h * self.Q_1 ** 0.5 / self.w * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5) + self.F_0:
            theta_2 = (self.F_1 - self.F_0) / (2 * (self.Q_1 ** 0.5 - self.Q_0 ** 0.5))
            theta_1 = self.h * self.Q_1 ** 0.5 / self.w
            N_0 = self.num_MUs * theta_2 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_2)
            G = self.num_MUs * (
                        (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) - (
                            1 / 3 * theta_1 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_1 / self.w))
            B_1 = self.num_MUs * (theta_1 - theta_2) * self.Q_1 / self.theta_m + self.num_MUs / self.theta_m * 1 / 3 * (
                        self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_1 ** 3 * self.w ** 2 / (self.h ** 2))
            m_total = G * (self.A - self.p * G) / 2 * self.num_EMs
            reward_0=N_0*self.F_0+self.c* B_1
            reward_1=N_1*self.F_1+self.p*m_total-self.c*B_1
            return reward_0,reward_1
        else:
            theta_3 = self.h / self.w * (
                        self.Q_0 ** 0.5 + (self.Q_0 - self.Q_1 - self.w / self.h * (self.F_0 - self.F_1)) ** 0.5)
            N_0 = self.num_MUs * theta_3 / self.theta_m
            N_1 = self.num_MUs * (1 - theta_3)
            G = self.num_MUs * (
                        (1 / 3 * self.theta_m ** 3 * self.w / (self.h ** 2) - self.Q_1 * self.theta_m / self.w) - (
                            1 / 3 * theta_3 ** 3 * self.w / (self.h ** 2) - self.Q_1 * theta_3 / self.w))
            B_1 = self.num_MUs / self.theta_m * 1 / 3 * (
                        self.theta_m ** 3 * self.w ** 2 / (self.h ** 2) - theta_3 ** 3 * self.w ** 2 / (self.h ** 2))
            m_total = G * (self.A - self.p * G) / 2 * self.num_EMs
            reward_0 = N_0 * self.F_0 + self.c * B_1
            reward_1 = N_1 * self.F_1 + self.p * m_total - self.c * B_1
            return reward_0, reward_1


