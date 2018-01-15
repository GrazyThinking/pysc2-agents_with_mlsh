from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib
import threading

from absl import app
from absl import flags
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf

from run_loop import run_loop

COUNTER = 0
SAVE_COUNTER=0
BOTH_UPDATE_COUNTER=0
MASTER_UPDATE_COUNTER=0
LOCK = threading.Lock()
FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_integer("warmup_length", 10, "Length of warmup period.")
flags.DEFINE_integer("joint_update_length", 20, "Length of joint update period.")
flags.DEFINE_integer("exp_record_length",64,"Length of replay collection")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 120, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 4, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)

if FLAGS.training:
    PARALLEL = FLAGS.parallel
    MAX_AGENT_STEPS = FLAGS.max_agent_steps
    DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
else:
    PARALLEL = 1
    MAX_AGENT_STEPS = 1e5
    DEVICE = ['/cpu:0']

LOG = FLAGS.log_path + FLAGS.map + '/' + FLAGS.net
SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)


def run_thread(agent, map_name, visualize, update_both=True, max_steps = FLAGS.max_steps):
    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        # Only for a single player!
        replay_buffer = []
        for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
            if FLAGS.training:
                replay_buffer.append(recorder)
                if is_done:
                    counter = 0
                    with LOCK:
                        global COUNTER
                        global SAVE_COUNTER
                        global BOTH_UPDATE_COUNTER
                        global MASTER_UPDATE_COUNTER
                        COUNTER += 1
                        SAVE_COUNTER+=1
                        counter = COUNTER
                        if update_both:
                            BOTH_UPDATE_COUNTER+=1
                        else:
                            MASTER_UPDATE_COUNTER+=1
                    # Learning rate schedule
                    learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
                    if update_both:
                        agent.update_master_policy(replay_buffer, FLAGS.discount, learning_rate, counter)
                        agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
                    else:
                        agent.update_master_policy(replay_buffer,FLAGS.discount, learning_rate, counter)
                    replay_buffer = []
                    if SAVE_COUNTER % FLAGS.snapshot_step == 1:
                        agent.save_model(SNAPSHOT, counter)
                    if counter >= max_steps:
                        break
            if is_done:
                obs = recorder[-1].observation
                score = obs["score_cumulative"][0]
                print('Agent have been trained for totally '+str(SAVE_COUNTER)+' times')
                print('Master have been updated '+str(MASTER_UPDATE_COUNTER)+' times')
                print('Both policies have been updated '+str(BOTH_UPDATE_COUNTER)+' times')
                print('Your score is ' + str(score) + '!')
        if FLAGS.save_replay:
            env.save_replay(agent.name)


def _main(unused_argv):
    """Run agents"""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.

    # Setup agents
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    print("agent name:" + agent_name)

    if agent_name == 'A3CAgent':
        agents = []
        for i in range(PARALLEL):
            agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
            agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
            agents.append(agent)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        summary_writer = tf.summary.FileWriter(LOG)
        for i in range(PARALLEL):
            agents[i].setup(sess, summary_writer)

        agent.initialize()
        if not FLAGS.training or FLAGS.continuation:
            global COUNTER
            COUNTER = agent.load_model(SNAPSHOT)

        # Run threads
        threads = []
        for i in range(PARALLEL - 1):
            t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False))
            threads.append(t)
            t.daemon = True
            t.start()
            time.sleep(5)

        run_thread(agents[-1], FLAGS.map, FLAGS.render)

        for t in threads:
            t.join()

        if FLAGS.profile:
            print(stopwatch.sw)

    elif agent_name == 'MLSHAgent':
        warmup_time = FLAGS.warmup_length
        joint_update_time = FLAGS.joint_update_length
        agents = []
        for i in range(PARALLEL):
            agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
            agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
            agents.append(agent)
        print('All agents built')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess_master = tf.Session(config=config)
        sess_subpolicies = tf.Session(config=config)
        summary_writer = tf.summary.FileWriter(LOG)
        for i in range(PARALLEL):
            agents[i].setup(sess_master,sess_subpolicies, summary_writer)
        agent.initialize()
        if not FLAGS.training or FLAGS.continuation:
            global COUNTER
            COUNTER = agent.load_model(SNAPSHOT)
        print('Starting training')
        for x in range(FLAGS.max_steps):
            print("Starting the "+str(x)+ "th training cycle")
            # agent.sample_env()
            agents[0].reset_master_policy()
            #Update master policy only
            threads = []
            for i in range(PARALLEL - 1):
                t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False,False,80))
                threads.append(t)
                t.daemon = True
                t.start()
                time.sleep(5)
            run_thread(agents[-1], FLAGS.map, FLAGS.render,False,80)
            for t in threads:
                t.join()
            COUNTER=0
            #Update both policies
            threads = []
            for i in range(PARALLEL - 1):
                t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False, True, 160))
                threads.append(t)
                t.daemon = True
                t.start()
                time.sleep(5)
            run_thread(agents[-1], FLAGS.map, FLAGS.render,True,160)
            for t in threads:
                t.join()
            COUNTER=0
            if FLAGS.profile:
                print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)
