# AHT_CrowdNav

## Setup
1. Install Python3.x
2. Install the required python package using pip or conda. For pip, use the following command:  
```
pip install -r requirements.txt
```
For conda, please install each package in `requirements.txt` into your conda environment manually and 
follow the instructions on the anaconda website.  

3. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.  


## Getting started
This repository is organized in three parts: 
- `crowd_sim/` folder contains the simulation environment. Details of the simulation framework can be found
[here](crowd_sim/README.md).
- `crowd_nav/` folder contains configurations and non-neural network policies
 
Below are the instructions for training and testing policies.

### Change configurations
1. Environment configurations and training hyperparameters: modify `crowd_nav/configs/config.py`


### Run the code

1. Run `check_env.py` to visualize the environment. The waypoint is selected randomly for now.
