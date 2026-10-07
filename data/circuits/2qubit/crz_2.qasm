OPENQASM 3.0;
include "stdgates.inc";
qubit[2] qs;
cx qs[1], qs[0];
t qs[1];
tdg qs[0];
cx qs[1], qs[0];
