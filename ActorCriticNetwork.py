import os
import re
import gym
import Config
import numpy as np
import tensorflow as tf
from threading import RLock
from multiprocessing import Value

class ActorCriticNetwork:

    def __init__(self, environments): 

        self.lock = RLock()

        self.actor_critics = dict()

        if Config.USE_EWC:
            self.fisher_batch = dict()
            self.consolidating = Value('b', False)

        if Config.PREPARE_EWC:
            self.fisher_batch = {}
            self.sampling = Value('b', False)
            self.sampling_finished = Value('b', False)

        self.graph = tf.Graph()

        with self.graph.as_default():

            with tf.device(Config.DEVICE):
                
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=Config.LEARNING_RATE,
                    decay=Config.RMS_DECAY,
                    momentum=Config.RMS_MOMENTUM,
                    epsilon=Config.RMS_EPSILON)

                self.build_input_network()

                self.environments = list(set(environments))

                if Config.USE_EWC: self.load_ewc_parameters()

                [self.build_actor_critic(environment) for environment in self.environments]

                self.initialize_backend()

    #############
    ### Model ###
    #############

    def build_input_network(self):
        self.input_layer = input_layer([None, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES], 'state_input')
        conv1, c1_w, c1_b = convolutional_layer_2d(self.input_layer, 8, 16, 'conv1', stride=4, activation_function=tf.nn.relu)
        conv2, c2_w, c2_b = convolutional_layer_2d(conv1, 4, 32, 'conv2', stride=2, activation_function=tf.nn.relu)
        flat = flatten(conv2)
        self.fc1, fc1_w, fc1_b = dense_layer(flat, 256, 'fully_connected', activation_function=tf.nn.relu)
        self.R = tf.placeholder(tf.float32, [None], name='R')
        self.shared_parameters = [c1_w, c1_b, c2_w, c2_b, fc1_w, fc1_b]

    def build_actor_critic(self, environment):

        if environment in self.actor_critics: return

        if Config.PREPARE_EWC: self.fisher_batch[environment] = []

        # My hacky way of getting the action space
        env = gym.make(environment)
        action_space = env.action_space.n
        env.close()

        input_layer, last_layer = self.input_layer, self.fc1

        actor_layer, _, _ = dense_layer(last_layer, action_space, 'actor_for_'+environment)
        critic_layer, _, _ = dense_layer(last_layer, 1, 'critic_for_'+environment)

        actor = tf.nn.softmax(actor_layer)
        critic = tf.squeeze(critic_layer, axis=[1])

        loss, actions_one_hot = self.setup_loss(actor, critic, action_space)

        minimizer = self.optimizer.minimize(loss)

        self.actor_critics[environment] = ActorCriticWrapper(actor, critic, minimizer, loss, self.R, actions_one_hot)

    def setup_loss(self, actor, critic, action_space):

        actions_one_hot = tf.placeholder(tf.float32, [None, action_space])

        action_probability = tf.reduce_sum(actor * actions_one_hot, axis=1)

        log_prob = tf.log(tf.maximum(action_probability, Config.LOG_NOISE))
        advantage = self.R - tf.stop_gradient(critic)
        entropy = tf.reduce_sum(tf.log(tf.maximum(actor, Config.LOG_NOISE)) * actor, axis=1)

        actor_loss = -(tf.reduce_sum((log_prob * advantage), axis=0) + tf.reduce_sum((-1 * Config.ENTROPY_BETA * entropy), axis=0))
        critic_loss = tf.reduce_sum(tf.square(self.R - critic), axis=0)

        loss = 0.5 * critic_loss + actor_loss

        if Config.USE_EWC and not Config.PLAY_MODE:

            F = self.elastic_weight_consolidation.F
            star_values = self.elastic_weight_consolidation.star_values
            current_parameters = self.shared_parameters

            for i, theta_i in enumerate(current_parameters):

                F_i = F[i].astype(np.float32)

                theta_star_i = star_values[i]

                sum_fisher = tf.reduce_sum(tf.multiply(F_i, tf.square(theta_i - theta_star_i)))

                loss += (Config.LAMBDA / 2) * sum_fisher

        return loss, actions_one_hot

    #######################
    ### Agent Interface ###
    #######################

    def train(self, environment, states, actions, rewards):

        if Config.PREPARE_EWC and self.sampling.value: self.fisher_batch[environment].append(states)

        actor_critic_wrapper = self.actor_critics[environment]
        minimizer = actor_critic_wrapper.minimizer
        R = actor_critic_wrapper.R
        actions_one_hot = actor_critic_wrapper.actions_one_hot

        with self.lock:
            self.session.run(minimizer, feed_dict={self.input_layer: states, R: rewards, actions_one_hot: actions})

    def predict(self, environment, states):
        actor_critic_wrapper = self.actor_critics[environment]
        actor = actor_critic_wrapper.actor
        critic = actor_critic_wrapper.critic
        with self.lock:
            policies, values = self.session.run([actor, critic], feed_dict={self.input_layer: states})
        return policies, values

    ###########
    ### EWC ###
    ###########

    def setup_EWC(self):

        print("Creating Fisher Information Matrix")

        with self.graph.as_default():
            with tf.device(Config.DEVICE):

                F = []
                N = 0

                for parameter in self.shared_parameters:
                    parameters_shape = parameter.get_shape().as_list()
                    parameter_fisher = np.zeros(parameters_shape)
                    F.append(parameter_fisher)

                for env, states in self.fisher_batch.items():

                    actor_critic_wrapper = self.actor_critics[env]
                    actor = actor_critic_wrapper.actor
                    action_index = tf.to_int32(tf.multinomial(tf.log(actor), 1)[0][0])

                    for state in states:
                        N += 1
                        derivatives = \
                            self.session.run(
                                tf.gradients(
                                    tf.log(actor[0, action_index]),
                                    self.shared_parameters
                                ),
                                feed_dict={self.input_layer: state})

                        for j in range(len(F)):
                            F[j] += np.square(derivatives[j])

                for i in range(len(F)):
                    F[i] /= N

                print(f"Created Fisher Information Matrix given {N} samples")

                return F

    def save_fisher_info_matrix(self, F, episode):
        savefile = '%s/%s_%08d_fisher.npy' % (Config.SAVE_DIRECTORY, Config.BRAIN, episode)
        np.save(savefile, F)

    def load_fisher_info_matrix(self):
        savefile = '%s/%s_%08d_fisher.npy' % (Config.SAVE_DIRECTORY, Config.BRAIN, Config.EWC_PREPARATION_EPISODES)
        F = np.load(savefile)
        return F

    def save_star_values(self, episode):
        star_values = []
        for parameter in self.shared_parameters:
            star_values.append(parameter.eval(session=self.session))
        savefile = '%s/%s_%08d_star.npy' % (Config.SAVE_DIRECTORY, Config.BRAIN, episode)
        np.save(savefile, star_values)

    def load_star_values(self):
        savefile = '%s/%s_%08d_star.npy' % (Config.SAVE_DIRECTORY, Config.BRAIN, Config.EWC_PREPARATION_EPISODES)
        star_values = np.load(savefile)
        return star_values

    def save_ewc(self, episode):
        F = self.setup_EWC()
        self.save_fisher_info_matrix(F, episode)
        self.save_star_values(episode)

    def load_ewc_parameters(self):
        print("> Using EWC to prevent catastrophic forgetting")
        try:
            F = self.load_fisher_info_matrix()
            star_values = self.load_star_values()
            self.consolidating.value = True
            self.elastic_weight_consolidation = ElasticWeightConsolidation(F, star_values)
            print("> Elastic Weight Consolidation Online")
        except:
            print("> No fisher info matrix found from previous network, skipping elastic weight consolidation but still sampling for next taskset")

    ###################
    ### Persistence ###
    ###################

    def save(self, episode):
        if Config.PLAY_MODE: return
        print(f"Saving {self.checkpoint_filename(episode)}")
        with self.lock:
            with self.graph.as_default():
                with tf.device(Config.DEVICE):
                    self.save_parameters(episode)
                    if Config.PREPARE_EWC and self.sampling_finished.value:
                        self.save_ewc(episode)
        print(f"Saved {self.checkpoint_filename(episode)}")

    def load(self):

        with self.graph.as_default():
            with tf.device(Config.DEVICE):
                self.update_checkpoint_file_brain(episode=Config.LOAD_EPISODE)
                filename = tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_filename(episode=Config.LOAD_EPISODE)))
                episode = int(re.split('/|_|\.', filename)[2])

                unknown_environments = self.recreate_graph(episode)
                self.load_parameters(filename, unknown_environments)

        return episode

    def recreate_graph(self, episode):
        known_environments = self.check_known_environments(episode)
        environments_not_in_session = list(filter(lambda env: env not in self.environments, known_environments))
        unknown_environments = list(filter(lambda env: env not in known_environments, self.environments))
        [self.build_actor_critic(environment) for environment in environments_not_in_session]
        self.environments.extend(environments_not_in_session)
        return unknown_environments

    def save_parameters(self, episode):
        variables = tf.global_variables()
        saver = tf.train.Saver({var.name: var for var in variables}, max_to_keep=0)
        saver.save(self.session, self.checkpoint_filename(episode))
        envs_filename = '%s/%s_%08d_envs' % (Config.SAVE_DIRECTORY, Config.BRAIN, episode)
        lines = []
        with open(envs_filename, "w") as f:
            for idx, env in enumerate(self.environments):
                if idx != len(self.environments):
                    lines.append(env + "\n")
                else:
                    lines.append(env)
            f.writelines(lines)

    def load_parameters(self, filename, unknown_environments):
        graph_parameters = tf.global_variables()
        saved_parameters = tf.train.NewCheckpointReader(filename).get_variable_to_shape_map()
        graph_parameters_to_load = {var.name: var for var in graph_parameters}
        for parameter in graph_parameters:
            if parameter.name not in saved_parameters.keys():
                new_environments_to_learn = \
                    [env_name for env_name in self.actor_critics
                     if env_name in parameter.name and env_name not in unknown_environments]
                if len(new_environments_to_learn) > 0:
                    unknown_environments.append(new_environments_to_learn[0])
                del graph_parameters_to_load[parameter.name]
        if len(unknown_environments) > 0:
            print(f"No Actor-Critics for {unknown_environments} found, spawned new policy-value layers")
        saver = tf.train.Saver(graph_parameters_to_load, max_to_keep=0)
        saver.restore(self.session, filename)

    def checkpoint_filename(self, episode):
        return '%s/%s_%08d' % (Config.SAVE_DIRECTORY, Config.BRAIN, episode)

    def update_checkpoint_file_brain(self, episode):

        filename = Config.SAVE_DIRECTORY+"/"+"checkpoint"

        with open(filename) as file:
            line1 = file.readline()
            line1_part2 = line1.split(" ")[1]

            if episode == 0:
                update_line = line1_part2.split("_")[1]
                update_episode = update_line.split("\"\n")[0]
                episode = int(update_episode)

            update = "%s_%08d" % (Config.BRAIN, episode)
            print(f"Attempting to load {update}")
            newline1 = "model_checkpoint_path: \"" + update + "\"\n"
            newline2 = "all_model_checkpoint_paths: \"" + update +"\"\n"

        with open(filename, "w") as file: file.writelines((newline1, newline2))
        Config.LOAD_EPISODE = episode
        print(f"Updated checkpoint to '{update}'")

    def check_known_environments(self, episode):

        save_file = '%s/%s_%08d_envs' % (Config.SAVE_DIRECTORY, Config.BRAIN, episode)

        with open(save_file) as f:

            lines = f.readlines()

            for idx, line in enumerate(lines):
                if line == "\n":
                    continue
                elif "\n" in line:
                    lines[idx] = line.split("\n")[0]

            known_environments = lines

            return known_environments

    #################
    ### Auxiliary ###
    #################

    def initialize_backend(self):
        self.session = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)))
        self.session.run(tf.global_variables_initializer())
        self.saver = None

####################
### Data Classes ###
####################

class ActorCriticWrapper:

    def __init__(self, actor, critic, minimizer, loss, R, actions_one_hot):
        self.actor = actor
        self.critic = critic
        self.minimizer = minimizer
        self.loss = loss
        self.R = R
        self.actions_one_hot = actions_one_hot

class ElasticWeightConsolidation:

    def __init__(self, F, star_values):
        self.F = F
        self.star_values = star_values

########################
### Tensorflow Utils ###
########################

def input_layer(shape, name):
    return tf.placeholder(tf.float32, shape, name)

def dense_layer(previous_layer, output_shape, name, activation_function=None):

    input_shape = previous_layer.get_shape().as_list()[-1]

    random_initializer = 1.0 / np.sqrt(input_shape)

    with tf.variable_scope(name):

        weight_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        bias_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        weights = tf.get_variable('w', dtype=tf.float32, shape=[input_shape, output_shape], initializer=weight_initializer)
        biases = tf.get_variable('b', shape=[output_shape], initializer=bias_initializer)

        dot_product = tf.matmul(previous_layer, weights) + biases

        output = activation_function(dot_product) if activation_function is not None else dot_product

    return output, weights, biases

def convolutional_layer_2d(previous_layer, filter_size, output_shape, name, stride, padding="SAME", activation_function=None):

    input_shape = previous_layer.get_shape().as_list()[-1]

    random_initializer = 1.0 / np.sqrt(filter_size * filter_size * input_shape)

    with tf.variable_scope(name):

        weight_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)
        bias_initializer = tf.random_uniform_initializer(-random_initializer, random_initializer)

        weights = tf.get_variable('w',
                                  shape=[filter_size, filter_size, input_shape, output_shape],
                                  dtype=tf.float32,
                                  initializer=weight_initializer)

        biases = tf.get_variable('b',
                                 shape=[output_shape],
                                 initializer=bias_initializer)

        convolution = tf.nn.conv2d(previous_layer, weights, strides=[1, stride, stride, 1], padding=padding) + biases

        output = activation_function(convolution) if activation_function is not None else convolution

    return output, weights, biases

def flatten(previous_layer):
    return tf.reshape(previous_layer, shape=[-1, (previous_layer.get_shape()[1] * previous_layer.get_shape()[2] * previous_layer.get_shape()[3])._value])
