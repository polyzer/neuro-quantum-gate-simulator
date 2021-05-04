import pandas as pd
import numpy as np
from queue import Queue
import pdb
import argparse
from tqdm import tqdm

import time
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
# from qiskit_textbook.tools import array_to_latex

from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, SGate, TGate, CXGate
from qiskit.visualization import plot_histogram

from collections import OrderedDict
from typing import List

parser = argparse.ArgumentParser(description='This script generates quantum gates dataset using Qiskit.')

parser.add_argument('--output_type', default='hdf', type=str)

args = parser.parse_args()

df = {
    "gate": [], # Name of Gate
    "qubits": [], # string of qubits
    "statevector": [], # Quantum state 
    "next_statevector": [],
    "unitary": [],
    "next_unitary": []
}

# init unitary simulator
unitary_backend = Aer.get_backend('unitary_simulator')
# init unitary simulator
statevector_backend = Aer.get_backend('statevector_simulator')

def get_all_qubits_pairs(qubits_count):
    '''
        generates all qubit pairs
    '''
    qubit_numbers = [i for i in range(qubits_count)]
    all_qubits_pairs = []
    for i in range(qubits_count):
        cur_qubit_num = qubit_numbers.pop()
        for j in qubit_numbers:
            all_qubits_pairs.append([cur_qubit_num, qubit_numbers[j]])
            all_qubits_pairs.append([qubit_numbers[j],cur_qubit_num])
    print("All qubits pairs generated")
    return all_qubits_pairs

"""
    ::arr:: list of objects
        {
            'gate': GateName(), // specified gate
            'qubits': [1]    // spedified qubits
        }
"""

def cliffordt_generator(qubits_count):
    gates = [XGate(), YGate(), ZGate(), HGate()]
    act_s = []

    for i in range(len(gates)):
        for j in range(qubits_count):
            act_s.append({"gate": gates[i], "qubits": [j]})
    all_qubits_pairs = get_all_qubits_pairs(qubits_count)
    for i in range(len(all_qubits_pairs)):
        act_s.append({"gate": CXGate(), "qubits": all_qubits_pairs[i]})  

    return act_s


"""
    ::arr:: list of objects
        {
            'gate': ZGate(), // specified gate
            'qubits': [1]    // spedified qubits
        }
"""
def generate_by_line(arr, qubits_count):
    qc = QuantumCircuit(qubits_count)
    for item in tqdm(arr):
        # pdb.set_trace()
        qubits = "".join(str(ch) for ch in item['qubits'])
        gate = item['gate'].name
        init_statevector = execute(qc,statevector_backend).result().get_statevector()
        init_unitary = execute(qc,unitary_backend).result().get_unitary()
        qc.unitary(item['gate'], item['qubits'])
        out_statevector = execute(qc,statevector_backend).result().get_statevector()
        out_unitary = execute(qc,unitary_backend).result().get_unitary()
        df['gate'].append(gate)
        df['qubits'].append(qubits)
        df['statevector'].append(init_statevector)
        df['next_statevector'].append(out_statevector)
        df['unitary'].append(init_unitary)
        df['next_unitary'].append(out_unitary)
        # pdb.set_trace()


# Count of qubits in quantumregister
qubits_count = 3
list_of_possible_actions = cliffordt_generator(qubits_count)
gates_lists = []
for i in range(100):
    le = []
    print(i)
    for j in range(1000):
        ri = np.random.randint(0, len(list_of_possible_actions))
        le.append(list_of_possible_actions[ri])
    gates_lists.append(le)

# inserting gates in their own queue
queues = []
end_queues = []
for i in range(len(gates_lists)):
    q = []
    for item in gates_lists[i]:
        q.append(item)
    queues.append(q)
    end_queues.append([])


# generation algorithm
while True:
    if len(queues[0]) == 0:
        break
    i = len(queues) - 1
    while True:
        # если указатель на последнем индексе!!!
        # то создаём линии
        if i == len(queues) - 1:
            for j in tqdm(range(len(queues[i]))):
                line = [item[0] for item in queues]
                generate_by_line(line, qubits_count)
                queues[i].append(queues[i].pop(0))
                # pdb.set_trace()
            i -= 1
            continue
        # if current is one of average index (!= 0 && != - 1)
        elif i != 0:
            #
            # if len(queues[i]) != 0:
            end_queues[i].append(queues[i].pop())
            if len(queues[i]) == 0:
                for _ in range(len(end_queues[i])):
                    queues[i].append(end_queues[i].pop())
                # come to i - 1 index to process it
                i -= 1
            else:
                # come to creating new lines
                break
            # else:
            #     # come to creating new lines
            #     break
        else:
            queues[0].pop()
            break

pd_df = pd.DataFrame(df)
if args.output_type == 'hdf':
    pd_df.to_hdf(f'quantum_dataset_{timestr}_output.h5', key='df')    
else:
    pd_df.to_csv(f'quantum_dataset_{timestr}_output.csv')