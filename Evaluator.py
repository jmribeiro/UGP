import sys

import Config
from UniversalGamePlayer import UniversalGamePlayer

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f"Usage:"
              f"\n\t$ python {sys.argv[0]} <environment> <first checkpoint> <last checkpoint> <interval>"
              f"\nExample: (Evaluates configured agent from 150000 training episodes to 200000 training episodes, each 10000)"
              f"\n\t$ python {sys.argv[0]} SpaceInvaders-v0 150000 200000 10000")
        sys.exit(-1)

    current_checkpoint = int(sys.argv[2])
    last_checkpoint = int(sys.argv[3])
    evaluation_interval = int(sys.argv[4])

    Config.PLAY_MODE = True
    Config.ENVIRONMENTS = sys.argv[1] * Config.PLAY_ACCELERATOR
    Config.TRAINERS = 0
    Config.LOAD = True

    while current_checkpoint <= last_checkpoint:

        print(f"Evaluating {Config.BRAIN} on {Config.ENVIRONMENTS[0]}: {current_checkpoint}/{last_checkpoint}")

        # 1 - Setup

        Config.EPISODES = 100
        Config.LOAD_EPISODE = current_checkpoint
        Config.PLAY_MODE = True
        Config.ENVIRONMENTS = [Config.ENVIRONMENTS[0]] * Config.PLAY_ACCELERATOR
        Config.TRAINERS = 0
        Config.LOAD = True

        agent = UniversalGamePlayer()

        # 2 - Run (and wait for it)

        agent.run()

        current_checkpoint += evaluation_interval
