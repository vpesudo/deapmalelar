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

ann_charge_isNeck_mcmvp_binaryclass.py
(note the vp) is a modification of the code including more data and calling from the command line for the number of epochs, and the number of neurons in each of the layers.
For instance: 

$ python3.5 ann_charge_isNeck_mcmvp_binaryclass.py 200 128 32 8

Trains an algorithm that trains 
200 epochs,
128 in the 1st hidden layer,
32 in the 2nd,
8 in the 3rd.

# Observations and lessons

It has been observed that increasing the sizeofbatch from 0.5 to 0.75 does not help in doing better predictions. All the models seem to reach the point where the test and train tendences diverge with epochs, pointing in the direction of lack of statistics.
