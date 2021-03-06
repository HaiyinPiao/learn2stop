import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time
from utils.args import *
import numpy as np



def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size):
    # def cat_s_a(s:torch.tensor, a:int):
    #     batch_size = 1
    #     label = torch.LongTensor([[a]])
    #     a = torch.zeros(batch_size, env.action_space.n).scatter_(1, label, 1)
    #     return torch.cat((s, a), 1)

    def cat_s_a_np(s:np.array, a:int):
        batch_size = 1
        # label = np.array([[a]])
        oh = np.zeros((batch_size, env.action_space.n))
        oh[0,a] = 1
        return np.append(s,oh)

    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        # repeat = 0
        # repeat_len = 0
        stop = True

        # batch_size = 1
        # label = torch.LongTensor(batch_size, 1).random_() % env.action_space.n
        # last_action = torch.zeros(batch_size, env.action_space.n).scatter_(1, label, 1)
        last_action = 0
        # ready_to_push = False
        # reward_period = 0
        state = cat_s_a_np(state, last_action)
        # interval = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0)
            # state_var = torch.cat((state_var, last_action), 1)
            # state_var = cat_s_a(state_var, 1)
            # Learn to stop, else maintain last action
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    # action, repeat = policy.select_action(state_var)[0].numpy()
                    action, stop = policy.select_action(state_var)
                    action = action[0].numpy()
                    stop = stop[0].numpy()
                    # print(action)
                    # print(repeat)
                    # exit()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            stop = int(stop)

            # action only updated when necessary.
            if t == 0 or stop is 1:
                last_action = action
            #     ready_to_push = True
            # repeat += 1

            assert(last_action is not None)
            next_state, reward, done, _ = env.step(last_action)

            next_state = cat_s_a_np(next_state, last_action)

            reward_episode += reward
            # reward_period += reward*(args.gamma**(repeat-1))

            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1
            memory.push(state, last_action, mask, next_state, reward, stop)
            # if ready_to_push == True or done:
            #     memory.push(state, last_action, mask, next_state, reward_period, stop, repeat)
            #     ready_to_push = False
            #     repeat = 0
            #     reward_period = 0 
                
            # memory.push(state, last_action, mask, next_state, reward, stop)

            if render:
                env.render()
            if done:
                # print(reward_episode)
                # print(t)
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
