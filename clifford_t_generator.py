import pandas as pd
import numpy as np
from queue import Queue
import pdb
import argparse
from tqdm import tqdm

import time

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
# from qiskit_textbook.tools import array_to_latex
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, SGate, TGate, CXGate
from qiskit.visualization import plot_histogram
from collections import OrderedDict
from typing import List

from utils import cliffordt_actions_generator

parser = argparse.ArgumentParser(description='This script generates quantum gates dataset using Qiskit.')

parser.add_argument('--output_type', default='hdf', type=str)

args = parser.parse_args()


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
            'gate': ZGate(), // specified gate
            'qubits': [1]    // spedified qubits
        }
"""
def generate_by_line(qc, item):
    gate_index = item['gate_index']
    init_statevector = execute(qc,statevector_backend).result().get_statevector()
    init_unitary = execute(qc,unitary_backend).result().get_unitary()
    qc.unitary(item['gate'], item['qubits'])
    out_statevector = execute(qc,statevector_backend).result().get_statevector()
    out_unitary = execute(qc,unitary_backend).result().get_unitary()
    df_item = {
        'gate_index': gate_index,
        'unitary': init_unitary,
        'next_unitary': out_unitary
    }
    return df_item
    # pdb.set_trace()


# Count of qubits in quantumregister
qubits_count = 3
list_of_possible_actions = cliffordt_actions_generator(qubits_count)
gates_lists = []
for i in range(20):
    le = []
    print(i)
    for j in range(1000):
        ri = np.random.randint(0, len(list_of_possible_actions))
        le.append(list_of_possible_actions[ri])
    gates_lists.append(le)

# inserting gates in their own queue
queues = []
for i in range(len(gates_lists)):
    q = []
    for item in gates_lists[i]:
        q.append(item)
    queues.append(q)


#generation algorithm
for q in tqdm(queues):
    df = []
    qc = QuantumCircuit(qubits_count)
    for j in tqdm(q):
        df.append(generate_by_line(qc, j))
    pd_df = pd.DataFrame(df)

    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.output_type == 'hdf':
        pd_df.to_hdf(f'datasets/quantum_dataset_{timestr}_output.h5', key='df')    
    else:
        pd_df.to_csv(f'datasets/quantum_dataset_{timestr}_output.csv')
