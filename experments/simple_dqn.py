# _*_ coding:utf-8 _*_
from collections import deque
import tensorflow as tf
import numpy as np
import random

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini batch


class SIMPLE_DQN():
    def __init__(self, env):
        # init replay buffer
        self.replay_buffer = deque()

        # init parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.state_input, self.q_value = self.create_q_network()
        self.action_input, self.y_input, self.cost, self.optimizer = self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_q_network(self):
        # network
        w1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        w2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        # input layer
        state_input = tf.placeholder("float", [None, self.state_dim])

        # hidden layer
        h_layer = tf.nn.relu(tf.matmul(state_input, w1) + b1)

        # Q value layer
        q_value = tf.matmul(h_layer, w2) + b2

        return state_input, q_value

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_training_method(self):
        action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        y_input = tf.placeholder("float", [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, action_input), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y_input - q_action))
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
        return action_input, y_input, cost, optimizer

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_q_network()

    def train_q_network(self):
        self.time_step += 1

        # step 1: obtain random mini batch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # step 2: calculate y
        y_batch = []
        q_value_batch = self.q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

        self.optimizer.run(feed_dict={self.y_input: y_batch,
                                      self.action_input: action_batch,
                                      self.state_input: state_batch})

    def egreedy_action(self, state):
        q_value = self.q_value.eval(feed_dict={self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value)

    def action(self, state):
        return np.argmax(self.q_value.eval(feed_dict={self.state_input: [state]})[0])
