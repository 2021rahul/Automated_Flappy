#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 100000.
EXPLORE = 2000000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

def createNetwork():

    s = tf.placeholder("float", [None, 80, 80, 4])

    w_conv1 = tf.Variable(tf.truncated_normal(shape=[8, 8, 4, 32],stddev=0.01))
    b_conv1 = tf.Variable(tf.constant(0.01,shape=[32]))
    conv = tf.nn.conv2d(s, w_conv1, [1, 4, 4, 1], padding="SAME")
    conv1 = tf.nn.relu(tf.nn.bias_add(conv, b_conv1))
    conv_pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    w_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 32, 64],stddev=0.01))
    b_conv2 = tf.Variable(tf.constant(0.01,shape=[64]))
    conv = tf.nn.conv2d(conv_pool1, w_conv2, [1, 2, 2, 1], padding="SAME")
    conv2 = tf.nn.relu(tf.nn.bias_add(conv, b_conv2))

    w_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64],stddev=0.01))
    b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))
    conv = tf.nn.conv2d(conv2, w_conv3, [1, 1, 1, 1], padding="SAME")
    conv3 = tf.nn.relu(tf.nn.bias_add(conv, b_conv3))
    conv3_flat = tf.reshape(conv3, [-1, 1600])

    w_fc1 = tf.Variable(tf.truncated_normal(shape=[1600,512],stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.01,shape=[512]))
    fc1 = tf.nn.relu(tf.matmul(conv3_flat, w_fc1) + b_fc1)

    w_fc2 = tf.Variable(tf.truncated_normal(shape=[512,ACTIONS],stddev=0.01))
    b_fc2 = tf.Variable(tf.constant(0.01,shape=[ACTIONS]))
    fc2 = tf.matmul(fc1, w_fc2) + b_fc2

    return s, fc2, fc1

def trainNetwork(s, fc2, fc1, sess):

    a = tf.placeholder("float", [None, ACTIONS])
    target = tf.placeholder("float", [None])
    prediction = tf.reduce_sum(tf.multiply(fc2, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(target - prediction))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()

    REPLAY = deque()

    INPUT_ACTION = np.zeros(ACTIONS)
    INPUT_ACTION[0] = 1
    x_t, r_0, terminal = game_state.frame_step(INPUT_ACTION)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_checkpoint")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = 00
    while "flappy bird" != "angry bird":
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                readout_t = fc2.eval(feed_dict={s : [s_t]})[0]
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        REPLAY.append((s_t, a_t, r_t, s_t1, terminal))
        if len(REPLAY) > REPLAY_MEMORY:
            REPLAY.popleft()

        if t > OBSERVE:

            minibatch = random.sample(REPLAY, BATCH)

            ss_batch = [d[0] for d in minibatch]
            aa_batch = [d[1] for d in minibatch]
            rr_batch = [d[2] for d in minibatch]
            ss1_batch = [d[3] for d in minibatch]

            readout_j1_batch = fc2.eval(feed_dict = {s : s_j1_batch})

            target_batch = []

            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    target_batch.append(r_batch[i])
                else:
                    target_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict = {
                target : target_batch,
                a : a_batch,
                s : s_j_batch}
            )

        s_t = s_t1
        t += 1

        if t % 10000 == 0:
            saver.save(sess, 'saved_checkpoint/' + GAME + '-dqn', global_step = t)

        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP:%d STATE:%s EPSILON:%d" % (t,state,epsilon))
        print("ACTION:%d REWARD:%d Q_MAX:%e" % (action_index,r_t,np.max(readout_t)))
        print()

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    s, fc2, fc1 = createNetwork()
    trainNetwork(s, fc2, fc1, sess)