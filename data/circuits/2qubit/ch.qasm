// Gate-count: 7, T-count: 2, time: 4.216s
OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[2];
sdg qubits[1];
h qubits[1];
tdg qubits[1];
cx qubits[0], qubits[1];
t qubits[1];
h qubits[1];
s qubits[1];
