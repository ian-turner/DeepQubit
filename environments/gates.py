import numpy as np


gates = {
    'h': {
        'cost': 1.0,
        'unitary': np.array([[1, 1],
                             [1, -1]], dtype=np.complex64) / np.sqrt(2),
    },
    's': {
        'cost': 1.0,
        'unitary': np.array([[1, 0],
                             [0, 1j]], dtype=np.complex64),
    },
    'sdg': {
        'cost': 1.0,
        'unitary': np.array([[1, 0],
                             [0, -1j]], dtype=np.complex64),
    },
    't': {
        'cost': 1.0,
        'unitary': np.array([[1, 0],
                             [0, np.exp(1j*np.pi/4)]], dtype=np.complex64),
    },
    'tdg': {
        'cost': 1.0,
        'unitary': np.array([[1, 0],
                             [0, np.exp(-1j*np.pi/4)]], dtype=np.complex64),
    },
    'x': {
        'cost': 1.0,
        'unitary': np.array([[0, 1],
                             [1, 0]], dtype=np.complex64),
    },
    'cx': {
        'cost': 1.0,
        'unitary': np.array([[0, 1],
                             [1, 0]], dtype=np.complex64),
        'controlled': True,
    },
    'y': {
        'cost': 1.0,
        'unitary': np.array([[0, -1j],
                             [1j, 0]], dtype=np.complex64),
    },
    'z': {
        'cost': 1.0,
        'unitary': np.array([[1, 0],
                             [0, -1]], dtype=np.complex64),
    },
}


def get_gate_set(gateset: str):
    gate_ids = gateset.split(',')
    return [{**gates[x], 'name': x} for x in gate_ids]
