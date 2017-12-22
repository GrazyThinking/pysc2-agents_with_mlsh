import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from agents.network import build_net
import utils as U

num_subpolicies = 3


class MLSHAgent(object):
    def __init__(self, training, msize, ssize, name='MLSH/MLSHAgent'):
        self.name = name
        self.training = training
        self.summary = []
        # Minimap size, screen size and info size
        assert msize == ssize
        self.msize = msize
        self.ssize = ssize
        self.isize = len(actions.FUNCTIONS)

    def setup(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer


    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def build_model(self, reuse, dev, ntype):
        with tf.variable_scope(self.name) and tf.device(dev):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Set inputs of networks
            self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize],
                                          name='minimap')
            self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

            # Build networks
            self.master_policy = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, num_subpolicies,
                                      'master_policy')
            subpolicies = []
            for i in range(num_subpolicies):
                subpolicy = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize,
                                      len(actions.FUNCTIONS), 'subpolicy')
                subpolicies.append(subpolicy)
            self.subpolicy_Q = self.master_policy
            self.spatial_action, self.non_spatial_action, self.value = subpolicies[tf.argmax(self.subpolicy_Q)]

            # Set targets and masks
            self.subpolicy_selected = tf.placeholder(tf.float32, [None, num_subpolicies], name='subpolicy_selected')
            self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                          name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                           name='valid_non_spatial_action')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                              name='non_spatial_action_selected')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

            # Compute log probability
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action,
                                                          axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

    def step(self, obs,warming_up):

        minimap = np.array(obs.observation['minimap'], dtype=np.float32)
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation['screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)

        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info: info}
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel()
        valid_actions = obs.observation['available_actions']
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        target = [int(target // self.ssize), int(target % self.ssize)]

        if False:
            print(actions.FUNCTIONS[act_id].name, target)

        # Epsilon greedy exploration
        if self.training and np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(valid_actions)
        if self.training and np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful
        return actions.FunctionCall(act_id, act_args)

    def update(self, rbs, disc, lr, cter):
        pass

    def save_model(self, path, count):
        self.saver.save(self.sess, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])

    def reset_master_policy(self):
        self.master_policy = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, num_subpolicies,
                                       'master_policy')
        self.subpolicy_Q = self.master_policy

    def update_master_policy(self):
        pass

    def update_subpolicies(self):
        pass
