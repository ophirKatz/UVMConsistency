# UVMConsistency
Accelerators final project. Proving UVM strong consistency fallouts


This program's goal is to prove that CUDA's
UVM consistency model is strong only when local
thread memory is flushed before handling page-
fault in CPU. And it shows that when this condition
is not adhered to, the consistency lacks strength.
       
The project showcases a real-life example of usage
of the UVM model, and presents a critical bug
caused by the consistency model issues.


The example at hand - a bank server, that receives
clients' requests that deposit an amount of money into
their account. The bug occurs when the balance is
later checked and the client discovers it has not
changed. The fix is to add the nessesary actions in
the code to insure the balance is correct.

