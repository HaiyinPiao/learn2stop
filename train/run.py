import argparse
import gym
import os
import sys
import pickle
import time
import datetime
import multiprocessing
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from utils.args import *
from utils.replay_memory import Memory
from utils.torch import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_ltr import LtrPolicy

try:
    path = os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name))
    models_file=open(path,'r')
    print("pre-trained models loaded.")
    print("model path: ", path)
except IOError:
    print("pre-trained models not found.")

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]+env.action_space.n
is_disc_action = len(env.action_space.shape) == 0
# running_state = ZFilter((state_dim,), clip=5)

"""seeding"""
seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

"""define actor and critic"""
if path is None:
    print("Can't find trained model!")
else:
    print(path)
    policy_net, value_net, _ = pickle.load(open(path, "rb"))

policy_net.to(device)
value_net.to(device)

state = env.reset()

stop = False
reward_episode = 0

def cat_s_a_np(s:np.array, a:int):
    batch_size = 1
    # label = np.array([[a]])
    oh = np.zeros((batch_size, env.action_space.n))
    oh[0,a] = 1
    return np.append(s,oh)

last_action = 0
state = cat_s_a_np(state, last_action)

for t in range(10000):
    state_var = tensor(state).unsqueeze(0)
    
    # assert(repeat>=0)
    # if repeat <= 0:
    with torch.no_grad():
        action, stop = policy_net.select_action(state_var)
        stop = bool(stop)
        print(stop)

    if stop is True or t==0:
        last_action = action

    next_state, reward, done, _ = env.step(int(last_action[0].numpy()))
    next_state = cat_s_a_np(next_state, last_action)
    reward_episode += reward

    if args.render is True:
        env.render()
        time.sleep(0.01)

    if done:
        print("reward:", reward_episode)
        break
    
    state = next_state