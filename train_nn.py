import pandas as pd
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

import time
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

df_name = 'quantum_dataset_2020-12-17_17-30-47_output.h5'


def unpack_complex_matrix_to_array_of_real_image_numbers_pairs(numpy_complex_matrix):
    flattened = numpy_complex_matrix.flatten()
    result = []
    for num in flattened:
        result.append(num.real)
        result.append(num.imag)
    return result

def concat_values(line):

    return line['statevector_unpacked'] + line['gate_onehot'] + line['qubits_onehot']

def load_dataset(df_name):
    """
        Loading and preprocessing data.
    """
    # load raw dataset
    df = pd.read_hdf(df_name)
    df['unitary_unpacked'] = df['unitary'].apply(unpack_complex_matrix_to_array_of_real_image_numbers_pairs)
    df['next_unitary_unpacked'] = df['next_unitary'].apply(unpack_complex_matrix_to_array_of_real_image_numbers_pairs)
    df['statevector_unpacked'] = df['statevector'].apply(unpack_complex_matrix_to_array_of_real_image_numbers_pairs)
    df['next_statevector_unpacked'] = df['next_statevector'].apply(unpack_complex_matrix_to_array_of_real_image_numbers_pairs)
    df['gate_onehot'] = pd.Series(LabelBinarizer().fit_transform(df.gate).tolist())
    df['qubits_onehot'] = pd.Series(LabelBinarizer().fit_transform(df.qubits).tolist())

    X = df[['gate_onehot', 'statevector_unpacked', 'qubits_onehot']].apply(concat_values, axis=1)
    Y = df['next_statevector_unpacked']
    # create dataset for translation [vectorstate, gatename,separator, qubits,separator] -> next_vectorstate
    # it defined qubits evolution.
    # pdb.set_trace()

    return [X, Y]



class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(input_shape, output_shape):
    # This is identical to the following:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(12,)))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dense(output_shape))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

[X, Y] = load_dataset(df_name)

input_shape = (None, len(X.iloc[0]))
output_shape = len(Y.iloc[0])

model = get_model(input_shape, output_shape)
# pdb.set_trace()
model.fit(np.array(X.to_list()), np.array(Y.to_list()), epochs=10, batch_size=128)