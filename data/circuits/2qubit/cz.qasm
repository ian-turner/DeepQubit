OPENQASM 3.0;
include "stdgates.inc";
qubit[2] qs;
h qs[0];
cx qs[1], qs[0];
h qs[0];
