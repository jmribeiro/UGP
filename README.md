## Setup Instructions

1) Install Nvidia's Cuda Toolkit [Tested on version 9.0]

2) Install OpenAI Gym:
    
    $ git clone https://github.com/openai/gym.git
    
    $ cd gym
    
    $ pip install -e .
    
    $ pip install -e '.[all]'

2) Install Requirements
    
    $ pip install tensorflow-gpu pandas
    
3) Unpack desired pre-trained agent from saves/freezer into saves [Default agent "UGP" already unpacked]

## Execution Instruction

1) Make configuration choices as desired using Config.py [Default configuration made for running the "UGP" agents on Space Invaders]

2) Run:

    $ python UniversalGamePlayer.py