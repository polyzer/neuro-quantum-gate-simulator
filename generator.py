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

"""
    ::arr:: list of objects
        {
            'gate': ZGate(), // specified gate
            'qubits': [1]    // spedified qubits
        }
"""
def generate_by_line(arr, qubits_count):
    qc = QuantumCircuit(qubits_count)
    for item in arr:
        qubits = "".join(item['qubits'])
        gate = item['gate'].name
        init_statevector = execute(qc,statevector_backend).result().get_statevector()
        init_unitary = execute(qc,unitary_backend).result().get_unitary()
        qc.unitary(item['gate'], item['qubits'])
        out_statevector = execute(qc,statevector_backend).result().get_statevector()
        out_unitary = execute(qc,unitary_backend).result().get_unitary()
        pdb.set_trace()
        



# inserting gates in their own queue
queues = []
end_queues = []
for i in range(len(gates_lists)):
    q = []
    for item in gates_lists[i]:
        q.append(item)
    queues.append(q)
    end_queues.append([])

# generation
while True:
    if len(queues[0]) == 0:
        break
    i = len(queues) - 1
    while True:
        # если указатель на последнем индексе!!!
        # то создаём линии
        if i == len(queues) - 1:
            for j in range(len(queues[i])):
                line = [item[0] for item in queues]
                queues[i].append(queues[i].pop(0))
                # pdb.set_trace()
            i -= 1
            continue
        # if current is one of average index (!= 0 && != - 1)
        elif i != 0:
            #
            if len(queues[i]) != 0:
                end_queues[i].append(queues[i].pop())
                if len(queues[i]) == 0:
                    for item in end_queues:
                        queues.append(item)
                    # come to i - 1 index to process it
                    i -= 1
                else:
                    # come to creating new lines
                    break
            else:
                # come to creating new lines
                break
        else:
            queues[0].pop()
            break
        



















        # 2. else
        if i == len(queues) - 1:
            for j in range(len(queues[i])):
                line = [item[0] for item in queues]
                end_queues[i].append(queues[i].pop(0))
                # pdb.set_trace()
        elif len(queues[i]) != 0:
            queues[i].append(queues[i].pop())
        else:
            queues[0].pop()