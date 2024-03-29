# Act-MARL

This code repository implements Act-MARL model, a Multi-Agent Model Based Reinforcement Learning Algorithm, that uses MAMBA ("Scalable Multi-Agent Model-Based Reinforcement Learning (https://arxiv.org/abs/2205.15023)) implementation of MAPPO. 

The repository contains Act-MARL implementation as well as fine-tuned hyperparameters in ```configs/dreamer/optimal``` folder.

To run the model, please refer to ```run.ipynb```, a google colab jupyter notebook, that can be used to run the model.

## Environment File

Download Starcraft zip files from 
* https://drive.google.com/file/d/1vJLclov1_WxvALS7yhRFtx6FkLcT8b27/view?usp=sharing
* https://drive.google.com/file/d/1BL4pExkrV6JQ2ADAQIcq8VddrdBqeMRo/view?usp=sharing


## Code Structure

- ```agent``` contains implementation of MAMBA 
  - ```controllers``` contains logic for inference
  - ```learners``` contains logic for learning the agent
  - ```memory``` contains buffer implementation
  - ```models``` contains architecture of MAMBA
  - ```optim``` contains logic for optimizing loss functions
  - ```runners``` contains logic for running multiple workers
  - ```utils``` contains helper functions
  - ```workers``` contains logic for interacting with environment
- ```env``` contains environment logic
- ```networks``` contains neural network architectures
