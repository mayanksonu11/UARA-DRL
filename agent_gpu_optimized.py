# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:36 2020
Modified for GPU optimization on 18/9/2025

@author: liangyu
@author: Cline (GPU optimizations)

Create the agent for a BS with GPU optimizations
"""

import copy
import numpy as np
from numpy import pi
from collections import namedtuple
from random import random, uniform, choice, randrange, sample
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from scenario import Scenario, BS

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple

class ReplayMemory(object):    # Define a replay memory

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DNN(nn.Module):  # Define a deep neural network

    def __init__(self, opt, sce, scenario):  # Define the layers of the fully-connected hidden network
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(sce.nUsers, 64)
        self.middle1_layer = nn.Linear(64, 32)
        self.middle2_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, sce.nUsers)
        
        # Initialize weights for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state):  # Define the neural network forward function
        x1 = F.relu(self.input_layer(state))
        x2 = F.relu(self.middle1_layer(x1))
        x3 = F.relu(self.middle2_layer(x2))
        out = self.output_layer(x3)
        return out
        

class BS_Agent:  # Define the agent (BS)

    def __init__(self, opt, sce, scenario, index, device):  # Initialize the agent (BS)
        self.opt = opt
        self.sce = sce
        self.id = index  # BS index
        self.device = device
        self.memory = ReplayMemory(opt.capacity)
        self.model_policy = DNN(opt, sce, scenario).to(device)
        self.model_target = DNN(opt, sce, scenario).to(device)
        self.model_target.load_state_dict(self.model_policy.state_dict())
        self.model_target.eval()
        self.optimizer = optim.RMSprop(params=self.model_policy.parameters(), lr=opt.learningrate, momentum=opt.momentum)
     
    def Select_Action(self, state, scenario, eps_threshold):   # Select action for a BS based on the network state
        sample = random()
        if sample < eps_threshold:  # epsilon-greedy policy : exploiting
            with torch.no_grad():
                # Ensure state is on correct device and has proper dimensions
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                state = state.to(self.device)
                Q_value = self.model_policy(state)   # Get the Q_value from DNN
                action = Q_value.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[randrange(self.sce.nUsers)]], dtype=torch.long, device=self.device)
        return action
        		
    def Get_Reward(self, action, action_i, state, scenario):  # action_i is UE selected by this BS
        ue_id = action_i.item() if hasattr(action_i, 'item') else action_i
        ue_loc = scenario.Get_UE_Location(ue_id)
        bs_loc = scenario.Get_BaseStations()[self.id].Get_Location()
        Loc_diff = bs_loc - ue_loc
        distance = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))
        Rx_power = scenario.Get_BaseStations()[self.id].Receive_Power(ue_id, distance)

        if Rx_power == 0.0:
            reward = self.sce.negative_cost
            Conn_State = 0
            layers = 0
        else:
            Interference = 0.0
            for j in range(self.opt.nagents):
                if action[j].item() == ue_id and j != self.id:  # Other BS also serving this UE
                    bs_j_loc = scenario.Get_BaseStations()[j].Get_Location()
                    Loc_diff_j = bs_j_loc - ue_loc
                    distance_j = np.sqrt((Loc_diff_j[0]**2 + Loc_diff_j[1]**2))
                    Rx_power_j = scenario.Get_BaseStations()[j].Receive_Power(ue_id, distance_j)
                    Interference += Rx_power_j
            Noise = 10**((scenario.Get_BaseStations()[self.id].Noise_dB)/10)
            SINR = Rx_power / (Interference + Noise)
            Rate = scenario.Get_BaseStations()[self.id].BS_Bw_Per_Channel * np.log2(1 + SINR) / (10**6)
            bs_req, el_req = scenario.Get_UE_Req_Rates(ue_id)
            if Rate >= bs_req:
                Conn_State = 1
                layers = 1 + (Rate - bs_req) // el_req
                reward = Rate
            else:
                Conn_State = 0
                layers = 0
                reward = self.sce.negative_cost
        reward = torch.tensor([reward], device=self.device)
        return Conn_State, reward, layers

    def Save_Transition(self, state, action, next_state, reward, scenario):  # Store a transition
        L = scenario.BS_Number()     # The total number of BSs
        K = self.sce.nChannel        # The total number of channels
        action = torch.tensor([[action]], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        # Ensure proper dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        self.memory.Push(state, action, next_state, reward)
    
    def Target_Update(self):  # Update the parameters of the target network
        self.model_target.load_state_dict(self.model_policy.state_dict())
            
    def Optimize_Model(self):
        if len(self.memory) < self.opt.batch_size:
            return
        transitions = self.memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors and move to device in batch
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool, device=self.device)
        
        # Batch process states
        state_batch = torch.cat([s.to(self.device) for s in batch.state])
        action_batch = torch.cat([a.to(self.device) for a in batch.action])
        reward_batch = torch.cat([r.to(self.device) for r in batch.reward])
        
        non_final_next_states = None
        if any(non_final_mask):
            non_final_next_states = torch.cat([s.to(self.device) for s in batch.next_state if s is not None])
        
        state_action_values = self.model_policy(state_batch).gather(1, action_batch)
        
        # Compute next state values
        next_state_values = torch.zeros(self.opt.batch_size, device=self.device)
        if non_final_next_states is not None and len(non_final_next_states) > 0:
            # Double DQN: Use policy network to select actions, target network to evaluate
            next_action_batch = self.model_policy(non_final_next_states).max(1)[1].detach()
            next_state_values[non_final_mask] = self.model_target(non_final_next_states).gather(1, next_action_batch.unsqueeze(1)).squeeze(1)
        
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model_policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Synchronize with target network less frequently for better GPU utilization
        # This is handled in main loop now
