import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import copy
import time 
import random
import numpy as np 

class A2CAgent(object):
    
    def __init__(self, actor, critic, args, env, dataGen):
        self.actor = actor
        self.critic = critic 
        self.args = args 
        self.env = env 
        self.dataGen = dataGen 
        print("agent is initialized")
        
    def train(self):
        args = self.args 
        env = self. env 
        dataGen = self.dataGen
        actor = self.actor 
        critic = self.critic 
        actor.train()
        critic.train()
        max_epochs = args['n_train']
        hidden_dim = args['hidden_dim']

        actor_optim = optim.Adam(actor.parameters(), lr=args['actor_net_lr'])
        critic_optim = optim.Adam(critic.parameters(), lr=args['critic_net_lr'])
        best_model = 1000
        r_test = []
        s_t = time.time()
        print("training started")
        
        for i in range(max_epochs):
            print("epch: ", i)
            data = dataGen.get_train_next()
            env.input_data = data 
            s, obs, avail_actions, s_coord, s_ch_l = env.reset()
            state_init = torch.from_numpy(s.reshape(env.batch_size, 1, env.n_nodes))
            static = env.input_pnt.astype(np.float32).reshape(env.batch_size, 2, env.n_nodes)
            static_ch_l = env.ch_l.astype(np.float32).reshape(env.batch_size, 1, env.n_nodes)
            actor.decoder.init_hidden(env.n_agents, env.batch_size)
            emb_static = actor.emd_stat(torch.from_numpy(static))
            critic.emb_stat(torch.from_numpy(static), torch.from_numpy(static_ch_l))
        
            current_time = np.zeros([env.batch_size]) 
            EV_vec = []
            charge_vec = []
            logprobs = []
            sh_vec = []
            sel_a = []
            for k in range(env.batch_size):
                sh_vec.append([])
                sel_a.append([])
                EV_vec.append([])
                charge_vec.append([])

            idx = torch.ones(env.batch_size, env.n_agents, 1).long()*(env.n_nodes-1)
            ter = torch.zeros(env.batch_size)
            terminated = np.zeros([env.batch_size])
            actions_all = []
            reward_all = []
            
            for step in range(args['decode_len']):
                actions = []
                logs = []
                for agent_id in range(env.n_agents):
                    static_coord = torch.from_numpy(s_coord[:, agent_id])
                    static_ch = torch.from_numpy(s_ch_l[:, agent_id])
                    decoder_input = torch.gather(emb_static, 2, idx[:, agent_id].view(-1, 1, 1).expand(env.batch_size, hidden_dim, 1))
                    dynamic = torch.from_numpy(obs[:, agent_id])
                    avail_actions[:, agent_id, env.n_nodes-1] += (np.sum(avail_actions[:, agent_id], axis =1) == np.zeros(env.batch_size)).astype(int)
                    av = torch.from_numpy(avail_actions[:, agent_id])
                    action, probs, logp = actor(static_coord, static_ch, dynamic, decoder_input, agent_id, ter, av)
                    new_av = copy.copy(avail_actions)
                    row = np.arange(env.batch_size)
                    avail_actions[row, :, action] = 0
                    col = np.ones(env.batch_size, dtype=int)*(env.n_nodes-1)
                    avail_actions[row, :, col] = new_av[row, :, col]
                    actions.append(action.view(env.batch_size, 1))
                    logs.append(logp.view(env.batch_size, 1))
    
    
                idx = torch.cat(actions, dim=-1)
                logs = torch.cat(logs, dim=-1)
                logprobs.append(logs.unsqueeze(0))
                s, obs, avail_actions, baseline, reward, terminated, padded, EV_vec, charge_vec, current_time, s_coord, s_ch_l, sel_a, sh_vec = env.step(idx, EV_vec, charge_vec, current_time, terminated, sel_a, sh_vec)
                ter = torch.from_numpy(terminated)
        
    

            logs = torch.cat(logprobs, dim=0).permute(1, 0, 2)
            critic_est = critic(state_init)
            R = current_time.astype(np.float32)
            print("R: ", R)
        
            R = torch.from_numpy(R)
            advantage = (R.view(-1, 1) - critic_est)
            actor_loss = torch.mean(advantage.detach() * logs.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args['max_grad_norm'])
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args['max_grad_norm'])
            critic_optim.step()
        
            e_t = time.time() - s_t
            print("episode: ", i)
            print("e_t: ", e_t)
            print("R: ", np.mean(current_time))
            if i % args['test_interval'] == 0:
        
                R = self.test(False)
                r_test.append(R)
                np.savetxt("trained_models/multi/test_rewards.txt", r_test)
            
                print("testing average rewards: ", R)
                if R < best_model:
                    best_model = R
                    num = str(i // args['save_interval'])
                    torch.save(actor.state_dict(), 'trained_models/multi' + '/' + 'best_model' + '_actor_params.pkl')
                    torch.save(critic.state_dict(), 'trained_models/multi' + '/' + 'best_model' + '_critic_params.pkl')
                
                
        
            if i % args['save_interval'] == 0:
                num = str(i // args['save_interval'])
                torch.save(actor.state_dict(), 'trained_models/multi' + '/' + num + '_actor_params.pkl')
                torch.save(critic.state_dict(), 'trained_models/multi' + '/' + num + '_critic_params.pkl')
            
    def test(self, inference):
        args = self.args 
        env = self. env 
        dataGen = self.dataGen
        actor = self.actor 
        critic = self.critic 
        actor.eval()
        hidden_dim = args['hidden_dim']
        data = dataGen.get_test_all()
        env.input_data = data 
        
        s, obs, avail_actions, s_coord, s_ch_l = env.reset() 
        static = env.input_pnt.astype(np.float32).reshape(env.batch_size, 2, env.n_nodes)
        static_ch_l = env.ch_l.astype(np.float32).reshape(env.batch_size, 1, env.n_nodes)
        actor.decoder.init_hidden(env.n_agents, env.batch_size)
        emb_static = actor.emd_stat(torch.from_numpy(static))
    
    
        sh_vec = []
        sel_a = []
        current_time = np.zeros([env.batch_size])
        EV_vec = []
        charge_vec = []
        sols = []
        for i in range(env.batch_size):
            sh_vec.append([])
            sel_a.append([])
            EV_vec.append([])
            charge_vec.append([])

        idx = torch.ones(env.batch_size, env.n_agents, 1).long()*(env.n_nodes-1)
        ter = torch.zeros(env.batch_size)
        terminated = np.zeros([env.batch_size])
        for step in range(args['decode_len']):
            actions = []
            for agent_id in range(env.n_agents):
                static_coord = torch.from_numpy(s_coord[:, agent_id])
                static_ch = torch.from_numpy(s_ch_l[:, agent_id])
                decoder_input = torch.gather(emb_static, 2, idx[:, agent_id].view(-1, 1, 1).expand(env.batch_size, hidden_dim, 1))
                dynamic = torch.from_numpy(obs[:, agent_id])
                av = torch.from_numpy(avail_actions[:, agent_id])
                action, probs, logp = actor(static_coord, static_ch, dynamic, decoder_input, agent_id, ter, av)
                new_av = copy.copy(avail_actions)
                row = np.arange(env.batch_size)
                avail_actions[row, :, action] = 0
                col = np.ones(env.batch_size, dtype=int)*(env.n_nodes-1)
                avail_actions[row, :, col] = new_av[row, :, col]
                actions.append(action.view(env.batch_size, 1))
    
            idx = torch.cat(actions, dim=-1)
            s_prime, obs_prime, avail_actions, baseline, reward, terminated, padded, EV_vec, charge_vec, current_time, s_coord, s_ch_l, sel_a, sh_vec = env.step(idx, EV_vec, charge_vec, current_time, terminated, sel_a, sh_vec)
            ter = torch.from_numpy(terminated)
            s = s_prime
            obs = obs_prime 
        
     
        actor.train()
        print("finished: ", sum(terminated))
        R = copy.copy(current_time)
        if inference:
            fname = 'test_results-{}-{}-len-{}-{}.txt'.format(args['test_size'], args['n_agents'], 
                                                               args['n_nodes'], args['difficulty'])
            fname = 'results/multi/' + fname
            np.savetxt(fname, R)
            
        return np.mean(R)