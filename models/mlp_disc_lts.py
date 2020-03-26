import torch.nn as nn
import torch
from utils.math import *

log_protect = 1e-5
multinomial_protect = 1e-10

BERNAULLICAT = 2

class DiscLtsPolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(200, 100), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        set_init(self.affine_layers)

        self.stop_hid = nn.Linear(last_dim, int(last_dim/2))
        self.action_hid = nn.Linear(last_dim, int(last_dim/2))
        set_init([self.stop_hid])
        set_init([self.action_hid])

        self.action_head = nn.Linear(int(last_dim/2), action_num)
        self.stop_head = nn.Linear(int(last_dim/2), BERNAULLICAT)
        set_init([self.action_head])
        set_init([self.stop_head])

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        ax = self.activation(self.action_hid(x))
        sx = self.activation(self.stop_hid(x))

        action_prob = torch.softmax(self.action_head(ax), dim=1)
        stop_prob = torch.softmax(self.stop_head(sx), dim=1)
        return action_prob, stop_prob

    def select_action(self, x):
        action_prob, stop_prob = self.forward(x)
        stop_prob += multinomial_protect
        stop = stop_prob.multinomial(1)
        action_prob += multinomial_protect
        action = action_prob.multinomial(1)
        return action, stop

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions, stops):
        action_prob, stop_prob = self.forward(x)
        action_prob = action_prob.gather(1, actions.long().unsqueeze(1))+log_protect
        stop_prob = stop_prob.gather(1, stops.long().unsqueeze(1))+log_protect

        # When stop==False, make action log(p)=0
        # print(stops)
        # print(action_prob)
        stops = stops.unsqueeze(1)
        action_prob = torch.where(stops == 1, action_prob, 1-stops)
        # print(action_prob)
        
        return torch.log(action_prob), torch.log(stop_prob)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

