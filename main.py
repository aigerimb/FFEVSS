import numpy as np 
import os 
import copy
import time 
import torch 
import random
from options import ParseParams


if __name__ == '__main__':
    args = ParseParams()   
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        print("# Set random seed to %d" % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    max_epochs = args['n_train']
    device = torch.device('cpu')
    if args['n_agents'] == 1:
        from neuralnets.nn_single  import Actor, Critic 
        from envs.EVCraft_single import DataGenerator, Env 
        from agents.agent_single import A2CAgent 
        critic = Critic(args['hidden_dim'])
        save_path = 'trained_models/single'
    else:
        from neuralnets.nn_multi  import Actor, Critic 
        from envs.EVCraft_multi import DataGenerator, Env 
        from agents.agent_multi import A2CAgent 
        critic = Critic(args['hidden_dim'], args['n_agents'])
        save_path = 'trained_models/multi'
    dataGen = DataGenerator(args)
    dataGen.reset()
    data = dataGen.get_train_next()
    env = Env(args, data)
    actor = Actor(args['hidden_dim'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        path = save_path +'/best_model_actor_params.pkl'
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, device))
            path = save_path + '/best_model_critic_params.pkl'
            critic.load_state_dict(torch.load(path, device))
            print("Succesfully loaded keys")
    
    agent = A2CAgent(actor, critic, args, env, dataGen)
    if args['train']:
        agent.train()
    else:
        R = agent.test(inference=True)
