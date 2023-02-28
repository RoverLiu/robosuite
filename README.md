## Build Status
PASS

## About

This package is modified from [robosuite official package](https://github.com/ARISE-Initiative/robosuite). An additionally package "PushAway" is added to the environment and the arena. 

"Push Away" aims to train an RL model which is capable to push the obstacle away with selected robot arm. More details for this environment could be found [here](https://github.com/RoverLiu/robosuite/blob/main/environments/manipulation/push_away.py).


## How to build
### Prerequisites
- Ubuntu 20.04 / macOS
- anaconda (recommended)
- Python 3 (Tested on python 3.9)
- mujoco
- numpy
- Pytorch

### Install anaconda

Install on macOS

https://docs.anaconda.com/anaconda/install/mac-os/

Install on ubuntu

https://docs.anaconda.com/anaconda/install/linux/

### Create a Development Workspace
``` bash
conda create --name myenv
```

### Install mujoco
Activate conda env
``` bash
conda activate myenv
```

Then, follow the process to install mjoco:

https://github.com/openai/mujoco-py

### Clone a Package
After setting up mujoco, robosuite can be installed with
``` bash
pip install robosuite
```

The following process is not a great solution, but it did solve the depency error. 
Navigate into the conda env
``` bash
cd anaconda3/envs/myenv/lib/python3.9/site-packages/robosuite
```

Replace all files there with the git repo [here](https://github.com/RoverLiu/robosuite)
``` bash
# delete everything here
git clone https://github.com/RoverLiu/robosuite
```

## How to test
Test your installation with:
``` bash
python -m robosuite.demos.demo_random_action
```

Ideally, the PushAway would show up when you select the environment.


## Dependencies

- mujoco
- robosuite
- numpy
- conda


