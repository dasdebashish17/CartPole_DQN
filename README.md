# CartPole_DQN
Implementing cartpole solution using DQN

Initially we implement the solution using Pytorch
For reference we have considered the DQN code provided in the official PyTorch page

[Reference_pytorch_implementation](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

Description of the files:
* [dqn_generic.py](src/dqn_generic.py) : Source code available in pytorch website
* [dqn_refactored.py](src/dqn_refactored.py) : Refactored the original code. Also added code to store the trained model and use it for playing 
* [parameter.json](src/parameter.json) : The epsilon and the corresponding return is logged here 
* [cartpole_dqn_model.mdl](src/cartpole_dqn_model.mdl) : Trained model
* [play_mode_statistics.log](src/play_mode_statistics.log) : How well the model has played after being trained.

We have 2 modes defined, TRAIN and PLAY
* TRAIN = 0
* PLAY = 1

Initially when we want to train, we set the mode to TRAIN
* mode = TRAIN

Later while replaying using the trained model, we can set the mode to PLAY
* mode = PLAY


*OBSERVATIONS*
To be updated
