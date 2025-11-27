from environments.qcircuit import *
from utils.matrix_utils import *


def path_to_qasm(path: List[QAction], num_qubits) -> str:
    qasm_str = ''
    qasm_str += 'OPENQASM 2.0;\n'
    qasm_str +='include "qelib1.inc";\n'
    qasm_str += 'qreg qubits[%d];\n' % num_qubits
    for x in path:
        name = x.name
        if x.name == 't10' or x.name == 't100':
            name = 't'
        qasm_str += '%s ' % name
        if isinstance(x, OneQubitGate):
            qasm_str += 'qubits[%d]' % x.qubit
        elif isinstance(x, ControlledGate):
            qasm_str += 'qubits[%d], qubits[%d]' % (x.control, x.target)
        qasm_str += ';\n'
    return qasm_str


def qasm_to_matrix(qasm_str: str) -> np.ndarray[np.complex128]:
    return Operator(qasm2.loads(qasm_str).reverse_bits()).data


def seq_to_matrix(seq: str) -> np.ndarray[np.complex128]:
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg qs[1];"""

    for x in seq:
        qasm_str += '\n' + x + ' qs[0];'
    
    qc = qasm2.loads(qasm_str)
    op = Operator.from_circuit(qc)
    return op.data
