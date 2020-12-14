import pandas as pd
import numpy as np
from queue import Queue
import pdb
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

gates_lists = [
    [
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
    ],
    [
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
    ],
    [
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
    ],
]

df = {
    "Gate": [], # Name of Gate
    "Qubits": [], # string of qubits
    "State": [], # Quantum state 
    "ResultState": []
}

# init unitary simulator
unitary_backend = Aer.get_backend('unitary_simulator')
# init unitary simulator
statevector_backend = Aer.get_backend('statevector_simulator')

def generate_by_line(arr):
    
    for item in arr:


# inserting gates in their own queue
queues = []
for i in range(len(gates_lists)):
    q = []
    for item in gates_lists[i]:
        q.append(item)
    queues.append(q)

# generation
while True:
    # for each index of gates_lists 
    cur_pointer = len(gates_lists) - 1
    if len(queues[0]) == 0:
        break

    for i in range(len(queues)):
        for _ in range(len(queues[i])):
            generate_by_line = []