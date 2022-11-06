# Ç!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time


def train(agent, env, evaluate):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    starttime = time.time()
    while step <= train_times:
        if step < args.warmup and step % 5 == 0:
            print("Step: {}, TrainTimes: {}".format(step, train_times))
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, 0)
        if step <= args.warmup:
            action = agent.select_action(observation, noise_factor=noise_factor)
        else:
            action = agent.select_action(observation, noise_factor=0)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)
        if episode_steps >= max_step and max_step:
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          '{:10d} Step_{:07d}/TrainTimes_{:07d}: mean_reward:{:.6f} mean_dist:{:.6f} var_dist:{:.6f}'.format(
                              int(time.time() - starttime), step - 1, train_times, np.mean(reward), np.mean(dist),
                              np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step > args.warmup:
                if step < 4000 * max_step:
                    #                    lr = (1e-4, 3e-4)
                    lr = (1e-4, 1e-4)
                elif step < 6000 * max_step:
                    #                    lr = (1e-4, 3e-4)
                    lr = (1e-4, 1e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    loss = agent.update_policy(lr)
                writer.add_scalar('train/actor_lr', lr[1], step)
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
        if step % 1000 == 0:
            agent.save_model(output, iter=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FontRL')

    # hyper-parameter
    parser.add_argument('--warmup', default=500, type=int,
                        help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=24, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=24, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=1, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0.5, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=50, type=int,
                        help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=8000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--font', default='2', type=str, help='font name')
    parser.add_argument('--run_id', default='1', type=str)
    args = parser.parse_args()
    # args.output = get_output_folder(args.output, "Paint")
    args.output = "./model/{}/Paint-run{}".format(args.font, args.run_id)
    print("Saving model in {}".format(args.output))
    print("Using bs={}".format(args.batch_size))
    writer = TensorBoard(args.output)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv

    fenv = fastenv(args.max_step, args.env_batch, writer, args.font)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output)
    evaluate = Evaluator(args, writer)
    train(agent, fenv, evaluate)
