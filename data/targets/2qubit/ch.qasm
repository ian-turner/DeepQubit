OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[2];
s qubits[0];
x qubits[0];
h qubits[0];
x qubits[1];
t qubits[0];
cx qubits[1], qubits[0];
h qubits[1];
t qubits[0];
s qubits[1];
s qubits[1];
h qubits[0];
h qubits[1];
x qubits[0];
z qubits[1];
s qubits[0];

