import pandas as pd
import numpy as np
from queue import Queue
import multiprocessing
import pdb

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

df = pd.DataFrame({
    "Gate": [],
    "Qubits": [],
    "State": [],
    "ResultState": []
})

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
            