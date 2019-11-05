# SFW

Code of the paper Stochastic In-Face Frank-Wolfe Methods for
Non-Convex Optimization and Sparse Neural
Network Training v1.0 (https://arxiv.org/pdf/1906.03580.pdf)

Comments w.r.t. to how the code was run. 

We used the Savio Cluster at the University of California, Berkeley, to run our experiments. The code is written in Pytorch and works both on cpu and gpu. For simplicity, we used ht_helper to run our code https://research-it.berkeley.edu/services/high-performance-computing/user-guide/hthelper-script

With ht_helper, if we had n independent jobs to run, we can send each job to a different core in a node (and queue jobs if needed). For our three experiments (Synthetic, MNIST and CIFAR10) we first have a python file in which we use ht_helper to search the hyperparameter space, and then a python file to re-run the experiments using the configuration with best validation value. For example, for the synthetic experiments, we try different steps sizes having a total of 360 different jobs when we use the Sigmoid activation functions (360 jobs between choosing different stepsizes and configurations). The file task file for that experiment looks like
/pathToFile/runSigmoid.py --id 0
..
/pathToFile/runSigmoid.py --id 359

While, when we re-run the experiments using the step size with the best validation loss, we run only 48 jobs for the Sigmoid experiment.

We organize the code as follows.
1) Files in the Main Folder. 
- OptimizerCode.py: Code of our Pytorch Optimizer. Our Optimizer allows to update the gradient of each layer using  SGD, Stochastic-Frank-Wolfe (SFW), AwaySteps, or iterating SFW and Away Steps independently.
- FrankWolfeRelated.py: Code for the implementation of SFW and the Away steps. For now, these methods only work in a dense layer.
- RunCode.py: It has the functions to train, validate and test a model, and a tailor-made train function for the method of SGD with thresholds. By default, in the train function, we calculate the validation and train error at the end of each epoch (and accuracy if required and possible). 
- Utilities.py:  This file contains functions that are used in the different experiments.

2) Models Folder.
- modelsSyntheticExp.py: Models for the synthetic experiment. SynRELU and SynSigmoid are the networks when we use RELU and Sigmoid as the activation functions resp.
- modelsMNIST.py: Models for the MNIST experiment. MNIST_Conv and MNIST_MLP are the convolutional and multi-layer perceptron networks we used resp.
- modelsCIFAR10.py: Model used for the CIFAR10.  This model is a shallow convolutional network. In the future, we want to try a state of the art architecture for the CIFAR10 experiment.

3) Synthetic Experiment Folder.
- CreateDataSynthetic.py: Executable file to create the data required for this experiment.
- ArtExpDataCreationFunctions.py: Auxiliary functions used in CreateDataSynthetic.py.
- runRELU.py/runSigmoid.py: Executable files to run the synthetic experiment using RELU and Sigmoid as activation functions resp. The tasks for each file go from --id 0 to --id 359. 
- runBestRELU.py/runBestSigmoid.py: Executable files. We use these files to re-run the experiments once the step-size with lowest validation error has been found. These python files save the network found at each epoch. The tasks go from --id 0 to --id 47. 

4) MNIST Experiment Folder.
- CreateDataMNIST.py: Executable file to create the data required for this experiment.
- runMLP.py/runConvOL.py: Executable files to run the MNIST experiment using the multi-layer perceptron and convolution networks. The 'OL' in the name of the convolutional file comes from the fact that we are only using Frank-Wolfe in one layer.  The latter differs from what was written in the paper. The tasks go from --id 0 to --id 34. 
- runBestMLP.py/runBestConvOL.py:  Executable files. We use these files to re-run the experiments once the step-size and configurations with the with lowest validation error has been found. In runMLP.py/runConvOL.py we used only one weight initialization to search for the best parameters, while here we try 30 different weight initializations.  These python files save the network found at each epoch. The tasks go from --id 0 to --id 89.

5) CIFAR10 Experiment Folder.
- CreateDataCIFAR10.py: Executable file to create the data required for this experiment.
- runConvSim.py: Executable file to run the CIFAR10 experiment using a convolution network. The tasks go from --id 0 to --id 34. 
- runBestConvSim.py: Executable file. We use this files to re-run the experiments once the step-size and configurations with the with lowest validation error has been found. In runConvSim.py we used only one weight initialization to search for the best parameters, while here we try 30 different weight initializations.  These python files save the network found at each epoch. The tasks go from --id 0 to --id 89.



