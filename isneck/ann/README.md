# Documentation on the different pieces of code in the folder
The idea of this folder is providing classification algorithms that allow us to identify background and physics events.
In this first phase we are focusing on IsNeck, classifying events as coming, or not, from the neck.

ann_charge_isNeck_mcm.py
It has several layers of hidden layers activated with relu functions, and a final layer with a sigmoid function.
Output is a real value between 0 and 1. One can set how this number converts to the categorical value (for instance, 
it can be set to 0 if val <0.5 and to 1 if val > 0.5.

ann_charge_isNeck_mcm_binaryclass.py
It has several layers of hidden layers activated with relu functions, and a final layer gives a categorical output 
(isNeckBackground. IsPhysics, etc).

