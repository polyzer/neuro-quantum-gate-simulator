import numpy as np
import pandas as pd
import pdb

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

class QuantumContinuousRLEnv(gym.Env):

    """
    Actions:
    There are 10 discrete deterministic actions:
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
    
    mode = "history"|"matrix"
    """
    def __init__(self, mode="history", observation_type = "discrete_history", actions_type = "discrete", max_trajectory_length = 6, qubits_count=2, action_gates_array = [], target_gates_array = []):
        super().__init__()
        
        self.target_gates_array = [
            {
                'gate': XGate(),
                'qubits': [0]
            },
            {
                'gate': XGate(),
                'qubits': [1]
            },            
        ]

        # init action gates array by parameters
        if target_gates_array:
            self.target_gates_array = target_gates_array

        self.action_gates_array = [
            {
                'gate': XGate(),
                'qubits': [0]
            },
            {
                'gate': XGate(),
                'qubits': [1]
            },
            {
                'gate': YGate(),
                'qubits': [0]
            },
            {
                'gate': YGate(),
                'qubits': [1]
            },
            {
                'gate': ZGate(),
                'qubits': [0]
            },
            {
                'gate': ZGate(),
                'qubits': [1]
            },
            {
                'gate': CXGate(),
                'qubits': [0, 1]
            },
            {
                'gate': CXGate(),
                'qubits': [1, 0]
            },
            {
                'gate': HGate(),
                'qubits': [0]
            },
            {
                'gate': HGate(),
                'qubits': [1]
            },

        ]

        # init action gates array by parameters
        if action_gates_array:
            self.action_gates_array = action_gates_array

        # qubits count in target circuit
        self.qubits_count = qubits_count

        self.mode = mode
        
        # actions shape
        self.num_actions = len(self.action_gates_array)
        
        # observation shape
        self.trajectory_length = max_trajectory_length
        
        
        # Gates history contains all applied gates in this trajectory
        self.gates_history = np.full((self.trajectory_length), 0)
        
        # maximum action value
        self.max_act = self.num_actions
        
        # threshold value
        self.treshold = 0.0001
        
        # action_space needs for RL algorithm
        self.action_space = gym.spaces.Discrete(self.num_actions)
        
        # init unitary simulator
        self.unitary_backend = Aer.get_backend('unitary_simulator')
        # init unitary simulator
        self.statevector_backend = Aer.get_backend('statevector_simulator')

    
        # get taget matrix, that algorithm must reconstruct
        self.target = self.get_target_matrix()
        self.linear_target = self.target.flatten()
        

        self.reset()
        
        # get 
        self.target_statevector = self.get_target_statevector()

        self.max_steps_count = 1000

        
        self.init_observation_space()

        self.get_reward = self.get_reward_by_history
        if self.mode != "history":
            self.get_reward = self.get_reward_by_matrix

    
    def init_observation_space(self):
        if self.mode == "history":
            self.init_box_history_observation_space()
        else:
            self.init_box_matrix_observation_space()        
    
    def init_box_history_observation_space(self):
        """
            Initialize observation space by
            current trajectory.
        """
        self.gates_history = np.full((self.trajectory_length), 0)
        self.observation_space = gym.spaces.Box(
            high=np.full(self.gates_history.shape, np.inf),
            low=np.full(self.gates_history.shape, -np.inf),
            dtype=np.float32
        )
        
    def init_box_matrix_observation_space(self):
        """
            Initialize observation space by
            current matrix.
        """
        self.current_matrix = np.full(self.get_target_matrix().shape, 0)
        self.current_flatten_matrix = self.current_matrix.flatten()
        self.observation_space = gym.spaces.Box(
            high=np.full(self.current_flatten_matrix.shape*2, np.inf),
            low=np.full(self.current_flatten_matrix.shape*2, -np.inf),
            dtype=np.float32
        )

    def init_current_statevector_end_statevector_observation_space(self):
        """
            Initialize observation space by
            current matrix.

            !!!!IS NOT CORRECT
        """
        self.current_matrix = np.full(self.get_target_matrix().shape, 0)
        self.current_flatten_matrix = self.current_matrix.flatten()
        self.observation_space = gym.spaces.Box(
            high=np.full(self.current_flatten_matrix.shape, np.inf),
            low=np.full(self.current_flatten_matrix.shape, -np.inf),
            dtype=np.float32
        )
    
    def reset(self):
        """
            Override gym.reset function.
        """
        qr = QuantumRegister(self.qubits_count, name="q")
        self.circuit = QuantumCircuit(qr)
        self.current_step = 0
        self.init_observation_space()
        obs = self.get_obs()
        return obs
    
    def get_target_matrix(self):
        """
        Returns matrix, that algorithm need to design
        """
        qc = QuantumCircuit(self.qubits_count)
        for gd in self.target_gates_array:
            qc.unitary(gd['gate'], gd['qubits'])
        unitary = execute(qc,self.unitary_backend).result().get_unitary()
        return unitary
    
    def get_target_statevector(self):
        return  execute(self.circuit, self.statevector_backend).result().get_statevector()

    def get_current_matrix(self):
        """
        Returns matrix, that algorithm build on current trajectory
        """

        unitary = execute(self.circuit, self.unitary_backend).result().get_unitary()
        return unitary
    
    def get_current_statevector(self):
        """
        Returns matrix, that algorithm build on current trajectory
        """
        
        stv = execute(self.circuit, self.statevector_backend).result().get_statevector()
        return stv


    def get_applied_gates_history(self):
        return self.gates_history


    def apply_gate_by_action(self, qc, action):
        """
        action: 0, max_action
        """
        # pdb.set_trace()
        gd = self.action_gates_array[action]
        qc.unitary(gd['gate'], gd['qubits'])
        self.gates_history[self.current_step] = action
        self.current_matrix = self.get_current_matrix()
        self.current_flatten_matrix = self.current_matrix.flatten()

    def apply_gate(self, qc, gate, qubits):
        qc.unitary(gate, *qubits)

    def get_obs(self):
        if self.mode == "history":
            return self.gates_history.copy()
        else:
            obs = self.current_flatten_matrix.real
            obs += self.current_flatten_matrix.imag
            return obs
        
    def render(self):
        print("Target Matrix: ")
        print(self.get_target_matrix())
        print("Current Matrix: ")
        print(self.get_current_matrix())
        
    def get_reward_by_history(self, cur_mat):
        reward = -1.0
        if (cur_mat == self.target).all():
            reward = 1.0
        return reward

    def get_reward_by_matrix(self, cur_mat):
        reward = -1.0
        value = np.abs(np.square(np.subtract(cur_mat,self.target)).sum())
        if value < self.treshold:
            reward = 1.0
            # pdb.set_trace()
        print("square: ", value)
        return reward

    """
    1. action -> gate
    2. apply gate
    3. reward
    """
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        #set reward by default
        reward = -1.0
        
        # set default done
        done = False
        
        self.apply_gate_by_action(self.circuit, action)
        
        cur_mat = self.get_current_matrix()
        
        if self.current_step == self.trajectory_length-1:
            done = True
                            
        reward = self.get_reward(cur_mat)
        if reward >= 0:
            done = True

        obs = self.get_obs()
        self.current_step += 1
        return  obs, reward, done, {}