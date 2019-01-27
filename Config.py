from EnvironmentAliases import *

"""
    Universal Game Player Configuration File
"""

EPISODES = 100                      # Number of episodes for the session

PLAY_MODE = True                    # Agent doesn't learn, acts only upon environment
PLAY_ACCELERATOR = 16               # Number of actor learners playing on the environment (Speeds up result gathering)
EXPLOITATION_DISTANCE = 0.10

BRAIN = "UGP"
ENVIRONMENTS = space_invaders       # Environments for the agent (See EnvironmentAliases)

RENDER_MODE = 1                     # OpenAI Gym Rendering (0=No Rendering; 1=Render agent 1; 2=Render all agents (slow)

LOAD = True                        	# Load from previous file
LOAD_EPISODE = 0                    # Load file episode (0 = Most recent)

SAVE_DIRECTORY = "saves"            # Directory where files are saved
SAVE_EPISODE_INTERVAL = 10000       # Episodes at which the model is saved

IMAGE_WIDTH = 84					# Resized screen width
IMAGE_HEIGHT = 84					# Resized screen height
STACKED_FRAMES = 4					# Lookback states the agent takes into account when making a decision

#######################
### HYPERPARAMETERS ###
#######################

LEARNING_RATE = 0.0003				# Learning Rate for Gradient Descent
DISCOUNT_FACTOR = 0.99				# Discount factor for reward accumulation

LOG_NOISE = 1e-6					# Minimum value for logarithmic computations
ENTROPY_BETA = 0.01					# Entropy weight for actor loss

REWARD_MIN_CLIP = -1                # All rewards below MIN_CLIP are set to MIN_CLIP
REWARD_MAX_CLIP = 1                 # All rewards above MAX_CLIP are set to MAX_CLIP

RMS_DECAY = 0.99
RMS_MOMENTUM = 0.0
RMS_EPSILON = 0.1

####################################
### Elastic Weight Consolidation ###
####################################

PREPARE_EWC = False                 # At the end of execution, sample for Fisher Info Matrix creation
SAMPLING_EPISODES = 50              # Total number of episodes for sampling

USE_EWC = False                     # Use Elastic Weight Consolidation for catastrophic forgetting alleviation
EWC_PREPARATION_EPISODES = 150000   # Checkpoint episodes for tasks to remember
LAMBDA = 100                        # How important old tasks are related to new tasks

#################################
### Asynchronous Architecture ###
#################################

DEVICE = 'gpu:0'                    # GPU device

PREDICTORS = 2                      # Number of threads processing prediction requests from actor-learners
PREDICTION_QUEUE_SIZE = 100         # Maximum prediction requests on queue
PREDICTION_MIN_BATCH = 128          # Number of predictions processed on GPU at a time

TRAINERS = 2                        # Number of threads processing datapoints from actor-learners
T_MAX = 5                           # No. of timesteps the actor-learner builds a recent experiences batch for training
TRAINING_QUEUE_SIZE = 100           # Maximum datapoints for training on queue
TRAINING_MIN_BATCH = 128            # Number of datapoints processed on GPU at a time

#############
### Stats ###
#############

STATS_FILE = SAVE_DIRECTORY+"/output-of-"+BRAIN+".csv"
