OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[3];
ccz qubits[0], qubits[1], qubits[2];
