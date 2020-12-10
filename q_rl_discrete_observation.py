import numpy as np
import pandas as pd

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
# from qiskit_textbook.tools import array_to_latex

from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate
from qiskit.visualization import plot_histogram

from collections import OrderedDict
from typing import List
import pdb
import gym
from gym import spaces
import gym


class QuantumDiscreteRLEnv(gym.Env):

    """
    Actions:
    There are 6 discrete deterministic actions:
    - 0: add X1
    - 1: add X2
    - 2: add Y1
    - 3: add Y2
    - 4: add Z1
    - 5: add Z2
    - 6: add CNOT12
    - 7: add CNOT21
    - 8: add H1
    - 9: add H2
    
    """
    def __init__(self, observation_type = "discrete_history", actions_type = "discrete", max_trajectory_length = 6):
        super().__init__()

        # actions shape
        self.num_actions = 10
        
        # observation shape
        self.trajectory_length = max_trajectory_length
        
        # all possible states count
        self.num_states = self.num_actions * self.trajectory_length
        
        # Gates history contains all applied gates in this trajectory
        self.gates_history = np.full((self.trajectory_length), 0)
        
        # maximum action value
        self.max_act = self.num_actions
        
        # threshold value
        self.treshold = 0.01
        
        # action_space needs for RL algorithm
        self.action_space = gym.spaces.Discrete(self.num_actions)
        
        # in current version observation is array of length = trajectory_length
        # where in each position we have one of numbers 0 ... 11
        self.observation_space_md = gym.spaces.MultiDiscrete(np.full((self.trajectory_length), 11).tolist())
        
        self.state = np.full((self.trajectory_length), 0)
        
        # init unitary simulator
        self.unitary_backend = Aer.get_backend('unitary_simulator')

        self.target = self.get_target_matrix()
        
        self.reset()
        
        self.max_steps_count = 1000
        
        self.observation_space_dimensions = self.observation_space_md.nvec
        self.observation_space = gym.spaces.Discrete(np.prod(self.observation_space_dimensions))

    def init_log():
        self.tryings_buffer = {
            'number': [],
            'trajectory_length': [],
            'observation': [],
            'action': []
        }

    def save_log():
        return
    def from_discrete_observation(self, observation: int) -> List[int]:
        """Convert a Discrete action to a MultiDiscrete action"""
        multi_observation = [None] * len(self.observation_space_dimensions)
        for idx, n in enumerate(self.observation_space_dimensions):
            observation, dim_observation = divmod(observation, n)
            multi_observation[idx] = dim_observation
        return multi_observation

    def to_discrete_observation(self, observation: int, base: int) -> List[int]:
        """Convert a MultiDiscrete action to a Discrete action"""
        out = 0
        for i in range(len(observation)):
            out += observation[i] * (base ** i) 
        return out
    
    def reset(self):
        qr = QuantumRegister(2, name="q")
        self.circuit = QuantumCircuit(qr)
        self.current_step = 0
        self.gates_history = np.full((self.trajectory_length), 0)
        return self.to_discrete_observation(self.gates_history.copy(), self.num_actions+1)
    
    def mse(self, A, B):
        mse = ((A - B)**2).sum()
        return mse
    
    """
    Returns matrix, that algorithm need to design
    """
    def get_target_matrix(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        unitary = execute(qc,self.unitary_backend).result().get_unitary()
        return unitary
    
    """
    Returns matrix, that algorithm build on current trajectory
    """
    def get_current_matrix(self):
        unitary = execute(self.circuit, self.unitary_backend).result().get_unitary()
        return unitary
    
    def get_applied_gates_history(self):
        return self.gates_history
    
    def apply_gate_by_action(self, qc, action):
        action += 1
        if action == 1:
            qc.x(0)
        elif action == 2:
            qc.x(1)
        elif action == 3:
            qc.y(0)
        elif action == 4:
            qc.y(1)
        elif action == 5:
            qc.z(0)
        elif action == 6:
            qc.z(1)
        elif action == 7:
            qc.cx(0, 1)
        elif action == 8:
            qc.cx(1, 0)
        elif action == 9:
            qc.h(0)
        elif action == 10:
            qc.h(1)
        self.gates_history[self.current_step] = action

    """
    1. action -> gate
    2. apply gate
    3. reward
    """
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        #set reward by default
        reward = -1.0
        
        self.state = (self.current_step, action)
        
        
        # set default done
        done = False
        
        self.apply_gate_by_action(self.circuit, action)
        
        cur_mat = self.get_current_matrix()
        
#         mse_value = self.mse(cur_mat, self.target)

        
        if self.current_step == self.trajectory_length-1:
            done = True
                                     
#         if mse_value <= self.treshold:
#             reward = 1.0
#             done = True
        if (cur_mat == self.target).all():
            reward = 1.0
            done = True

        print("Target Matrix: ")
        print(self.get_target_matrix())
        print("Current Matrix: ")
        print(self.get_current_matrix())
        obs = self.to_discrete_observation(self.gates_history.copy(), self.num_actions+1)
        self.current_step += 1
        return  obs, reward, done, {}