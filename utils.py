from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, SGate, TGate, CXGate

"""
    ::arr:: list of objects
        {
            'gate': GateName(), // specified gate
            'qubits': [1]    // spedified qubits
        }
"""

def cliffordt_actions_generator(qubits_count):
    gates = [XGate(), YGate(), ZGate(), HGate()]
    act_s = []

    for i in range(len(gates)):
        for j in range(qubits_count):
            act_s.append({"gate": gates[i], "qubits": [j]})
    all_qubits_pairs = get_all_qubits_pairs(qubits_count)
    for i in range(len(all_qubits_pairs)):
        act_s.append({"gate": CXGate(), "qubits": all_qubits_pairs[i]})  
    
    for i in range(len(act_s)):
        act_s[i]['gate_index'] = i
    return act_s


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