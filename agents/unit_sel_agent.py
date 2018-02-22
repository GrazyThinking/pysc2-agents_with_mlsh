import random

import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from s2clientprotocol import ui_pb2 as sc_ui

import utils as U
from agents.network import build_net

num_units = 2
batch_size = 64

class UnitSelAgent(object):
    def __init__(self, training, msize, ssize, name='UNITSEL/UnitSelAgent'):
        self.name = name
        self.training = training
        self.summary = []
        # Minimap size, screen size and info size
        assert msize == ssize
        self.msize = msize
        self.ssize = ssize
        self.isize = len(actions.FUNCTIONS)
        # Be careful, workaround
        self.init_counter = 0

    def setup(self, sess_master, summary_writer, sess_subpolicies=None, ):
        self.sess_master = sess_master
        # self.sess_subpolicies = sess_subpolicies
        self.summary_writer = summary_writer

    def reset(self):
        # Epsilon schedule
        self.epsilon = [0.05, 0.2]

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess_master.run(init_op)
        # self.sess_subpolicies.run(init_op)

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

            # create master and subpolicies
            self.subpolicy_Q = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize,
                                         num_units + 2, 'master_policy')

            # Set targets and masks for master policy update
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')

            self.action_input = tf.placeholder("float", [None, num_units + 2])
            self.y_input = tf.placeholder("float", [None])
            self.Q_action = tf.reduce_sum(tf.multiply(self.subpolicy_Q, self.action_input), reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.Q_action))
            self.master_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            # Set targets and masks for subpolicies update
            self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action_')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                          name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                           name='valid_non_spatial_action_')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                              name='non_spatial_action_selected_')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target_')

            # Build the optimizer
            opt = tf.train.AdamOptimizer(self.learning_rate)

            self.subpolicy = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize,
                                       len(actions.FUNCTIONS), 'fcn')
            self.spatial_action, self.non_spatial_action, self.value = self.subpolicy

            # Compute log probability
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action,
                                                          axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('spatial_action_prob_', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob_', non_spatial_action_prob))

            # Compute losses, more details in https://arxiv.org/abs/1602.01783
            # Policy loss and value loss
            action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.value_target - self.value)
            policy_loss = - tf.reduce_mean(action_log_prob * advantage)
            value_loss = - tf.reduce_mean(self.value * advantage)

            self.summary.append(tf.summary.scalar('policy_loss_', policy_loss))
            self.summary.append(tf.summary.scalar('value_loss_', value_loss))

            # TODO: policy penalty
            loss = policy_loss + value_loss

            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads:
                # get around of master policy gradients
                if grad is None:
                    continue
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)
            self.summary_op = tf.summary.merge(self.summary)

            self.saver = tf.train.Saver(max_to_keep=100)

    def step(self, obs, use_unit_selector):
        minimap = np.array(obs.observation['minimap'], dtype=np.float32)
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation['screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1
        if self.init_counter==0:
            self.init_counter+=1
            return actions.FunctionCall(7,[[1]])
        elif self.init_counter==1:
            self.init_counter+=1
            return actions.FunctionCall(5,[[sc_ui.ActionMultiPanel.SingleSelect], [0]])
        elif self.init_counter==2:
            self.init_counter+=1
            return actions.FunctionCall(4,[[1],[0]])
        elif self.init_counter==3:
            self.init_counter+=1
            return actions.FunctionCall(7, [[1]])
        elif self.init_counter==4:
            self.init_counter+=1
            return actions.FunctionCall(5, [[sc_ui.ActionMultiPanel.SingleSelect], [1]])
        elif self.init_counter==5:
            self.init_counter+=1
            return actions.FunctionCall(4,[[1],[1]])
        elif use_unit_selector:
            unitSel = self.get_unit_sel_res(obs)
            if self.training and np.random.rand() < self.epsilon[0]:
                unitSel = np.random.randint(0, 4)
            if unitSel == num_units + 1:
                return actions.FunctionCall(7, [[1]])
            elif unitSel == num_units:
                feed = {self.minimap: minimap,
                        self.screen: screen,
                        self.info: info}

                non_spatial_action, spatial_action = self.sess_master.run(
                    [self.non_spatial_action, self.spatial_action],
                    feed_dict=feed)

                # Select an action and a spatial target
                non_spatial_action = non_spatial_action.ravel()
                spatial_action = spatial_action.ravel()
                valid_actions = obs.observation['available_actions']
                act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
                target = np.argmax(spatial_action)
                target = [int(target // self.ssize), int(target % self.ssize)]

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
            else:
                return actions.FunctionCall(4, [[0], [unitSel]])
        else:
            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}

            non_spatial_action, spatial_action = self.sess_master.run(
                [self.non_spatial_action, self.spatial_action],
                feed_dict=feed)

            # Select an action and a spatial target
            non_spatial_action = non_spatial_action.ravel()
            spatial_action = spatial_action.ravel()
            valid_actions = obs.observation['available_actions']
            act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
            target = np.argmax(spatial_action)
            target = [int(target // self.ssize), int(target % self.ssize)]

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

    def reset_init_counter(self):
        self.init_counter=0

    def save_model(self, path, count):
        self.saver.save(self.sess_master, path + '/model.pkl', count)
        # self.saver.save(self.sess_subpolicies, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mconv1_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mconv2_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sconv1_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sconv2_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='info_fc_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='spatial_action_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feat_fc_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='non_spatial_action_sub_None') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value_sub_None')
        partial_saver = tf.train.Saver(var_list=var_list)
        partial_saver.restore(self.sess_master, ckpt.model_checkpoint_path)
        # self.saver.restore(self.sess_subpolicies, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])

    def reset_master_policy(self):
        master_variables = []
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mconv1_master'))
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mconv2_master'))
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sconv1_master'))
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sconv2_master'))
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='info_fc_master'))
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feat_fc_master'))
        master_variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='subpolicy_Q'))
        for i in master_variables:
            master_init_op = tf.variables_initializer(i)
            self.sess_master.run(master_init_op)

    def update_unitsel(self, rbs, disc, lr, cter):
        samples = random.sample(rbs, batch_size)
        minimaps = []
        screens = []
        infos = []
        next_minimaps = []
        next_screens = []
        next_infos = []
        actions = []
        rewards = []
        for i, [obs, _, action, _, next_obs] in enumerate(samples):
            minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            next_minimap = np.array(next_obs.observation['minimap'], dtype=np.float32)
            next_minimap = np.expand_dims(U.preprocess_minimap(next_minimap), axis=0)
            next_screen = np.array(obs.observation['screen'], dtype=np.float32)
            next_screen = np.expand_dims(U.preprocess_screen(next_screen), axis=0)
            next_info = np.zeros([1, self.isize], dtype=np.float32)
            next_info[0, obs.observation['available_actions']] = 1
            reward = next_obs.reward

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)
            next_minimaps.append(next_minimap)
            next_screens.append(next_screen)
            next_infos.append(next_info)
            cur_action = np.zeros(num_units + 2)
            cur_action[action] = 1
            actions.append(cur_action)
            rewards.append(reward)

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)
        next_minimaps = np.concatenate(next_minimaps, axis=0)
        next_screens = np.concatenate(next_screens, axis=0)
        next_infos = np.concatenate(next_infos, axis=0)
        y_batch = []
        Qvalue_batch = self.sess_master.run(self.subpolicy_Q, feed_dict={self.minimap: next_minimaps,
                                                                         self.screen: next_screens,
                                                                         self.info: next_infos})
        for i in range(0, batch_size):
            terminal = samples[i][3]
            if terminal:
                y_batch.append(rewards[i])
            else:
                y_batch.append(rewards[i] + disc * np.max(Qvalue_batch[i]))

        self.sess_master.run(self.master_train_op, feed_dict={self.minimap: minimaps,
                                                              self.screen: screens,
                                                              self.info: infos,
                                                              self.y_input: y_batch,
                                                              self.action_input: actions,
                                                              self.learning_rate: lr})

    def update_main_policy(self, rbs, disc, lr, cter):
        # Compute R, which is value of the last observation
        obs = rbs[-1][-1]
        if obs.last():
            R = 0
        else:
            minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            _, _, value = self.subpolicy
            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}
            R = self.sess_master.run(value, feed_dict=feed)
        # Compute targets and masks
        minimaps = []
        screens = []
        infos = []

        value_target = np.zeros([len(rbs)], dtype=np.float32)
        value_target[-1] = R

        valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
        spatial_action_selected = np.zeros([len(rbs), self.ssize ** 2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

        rbs.reverse()
        for i, [obs, action, unit_sel_res, _, next_obs] in enumerate(rbs):
            if unit_sel_res!=num_units:
                continue
            minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)

            reward = obs.reward
            act_id = action.function
            act_args = action.arguments

            value_target[i] = reward + disc * value_target[i - 1]

            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1
        if len(minimaps)!=0 and len(screens)!=0 and len(infos)!=0:
            minimaps = np.concatenate(minimaps, axis=0)
            screens = np.concatenate(screens, axis=0)
            infos = np.concatenate(infos, axis=0)

            # Train
            feed = {self.minimap: minimaps,
                    self.screen: screens,
                    self.info: infos,
                    self.value_target: value_target,
                    self.valid_spatial_action: valid_spatial_action,
                    self.spatial_action_selected: spatial_action_selected,
                    self.valid_non_spatial_action: valid_non_spatial_action,
                    self.non_spatial_action_selected: non_spatial_action_selected,
                    self.learning_rate: lr}
            _, summary = self.sess_master.run([self.train_op, self.summary_op], feed_dict=feed)
            self.summary_writer.add_summary(summary, cter)

    def isMLSH(self):
        return True

    def get_unit_sel_res(self, obs):
        minimap = np.array(obs.observation['minimap'], dtype=np.float32)
        minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation['screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        feed_master = {self.minimap: minimap,
                       self.screen: screen,
                       self.info: info}
        subpolicy_selected = np.argmax(self.sess_master.run(self.subpolicy_Q, feed_dict=feed_master), axis=1)[0]
        return subpolicy_selected
