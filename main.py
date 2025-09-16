# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:49:45 2020

@author: liangyu

Running simuluation
"""

import copy, json, argparse
import torch
from scenario import Scenario
from agent import BS_Agent
from dotdic import DotDic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_agents(opt, sce, scenario, device):
	agents = []   # Vector of agents
	for i in range(scenario.BS_Number()):
		agents.append(BS_Agent(opt, sce, scenario, index=i, device=device)) # Initialization, create a CNet for each agent
	return agents
    
def run_episodes(opt, sce, agents, scenario): 
    global_step = 0
    nepisode = 0
    final_reward = 0
    final_layers = 0
    total_connected = 0

    action = torch.zeros(opt.nagents,dtype=int)
    reward = torch.zeros(opt.nagents)
    Conn_State = torch.zeros(sce.nUsers)
    state_target = torch.ones(sce.nUsers)  # The Conn_State requirement
    layers = torch.zeros(opt.nagents, dtype=int)  # Number of layers for each agent
    f= open("DDQN.csv","w+")
    f.write("Episode,Num_BaseStations,Num_Users,Final_Reward,Total_Connected,Final_Layers\n")
    while nepisode < opt.nepisodes:
        state = torch.zeros(sce.nUsers)  # Reset the state
        next_state = torch.zeros(sce.nUsers)  # Reset the next_state
        nstep = 0
        while nstep < opt.nsteps:
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max  # Linear increasing epsilon
                # eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-1. * nstep * (nepisode + 1)/opt.eps_decay)
                # Exponential decay epsilon
            for i in range(opt.nagents):
                action[i] = agents[i].Select_Action(state, scenario, eps_threshold)  # Select action
            Conn_State = torch.zeros(sce.nUsers)
            for i in range(opt.nagents):
                ue_id = action[i].item()
                Conn_State[ue_id], reward[i], layers[i] = agents[i].Get_Reward(action, action[i], state, scenario)  # Obtain reward and next state
            next_state = Conn_State.clone()
            for i in range(opt.nagents):
                # print('Agent:', i, 'Action:', action[i], 'Reward:', reward[i], 'Conn_State:', Conn_State[i])
                agents[i].Save_Transition(state, action[i], next_state, reward[i], scenario)  # Save the state transition
                agents[i].Optimize_Model()  # Train the model
                if nstep % opt.nupdate == 0:  # Update the target network for a period
                    agents[i].Target_Update()
            state = next_state.clone()  # State transits
            if torch.all(state.eq(state_target)):  # If Conn_State is satisified, break
                break
            nstep += 1
        print('Episode Number:', nepisode, 'Training Step:', nstep)       
        final_reward = torch.sum(reward).item()  # Sum the reward
        print('Final Reward:', final_reward)
        print('Total Layers:', layers)
        final_layers = layers
        print('Total connected UEs:', sum(state))
        total_connected = torch.sum(state).item()  # Sum the Conn_State
        # f.write("%i \n" % nstep)
        f.write(str(nepisode) + "," + str(sce.nMBS + sce.nSBS  + sce.nFBS) + ","  + str(sce.nUsers) + "," + str(final_reward) + "," + str(total_connected) + "," + str(final_layers.tolist()) +  "\n")
        nepisode += 1
    f.close()
                
def run_trial(opt, sce):
    scenario = Scenario(sce)
    # print(len(scenario.BaseStations))
    agents = create_agents(opt, sce, scenario, device)  # Initialization 
    run_episodes(opt, sce, agents, scenario)    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_path1', type=str, help='path to existing scenarios file')
    parser.add_argument('-c2', '--config_path2', type=str, help='path to existing options file')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()
    sce = DotDic(json.loads(open(args.config_path1, 'r').read()))
    opt = DotDic(json.loads(open(args.config_path2, 'r').read()))  # Load the configuration file as arguments
    print(sce)
    print(opt)
    assert(opt.nagents == sce.nMBS + sce.nSBS + sce.nFBS)
    for i in range(args.ntrials):
        trial_result_path = None
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt, trial_sce)
