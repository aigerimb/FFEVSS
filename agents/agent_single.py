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

        actor_optim = optim.Adam(actor.parameters(), lr=args['actor_net_lr'])
        critic_optim = optim.Adam(critic.parameters(), lr=args['critic_net_lr'])
        best_model = 1000
        r_test = []
        s_t = time.time()
        print("training started")
        for i in range(max_epochs):
            
            data = dataGen.get_train_next()
            env.input_data = data 
            state, avail_actions = env.reset()
            static = env.input_data[:, :, :2].reshape(env.batch_size, 2, env.n_nodes).astype(np.float32)
            ch_l = env.ch_l.reshape(env.batch_size, 1, env.n_nodes).astype(np.float32)
            EV_vec = []
            charge_vec = []
            for j in range(env.batch_size):
                EV_vec.append([])
                charge_vec.append([])
            current_time = np.zeros([env.batch_size])
            static_hidden, static_ch_l = actor.emd_stat(torch.from_numpy(static), torch.from_numpy(ch_l))
            dynamic_init = env.d.reshape(env.batch_size, 1, env.n_nodes).astype(np.float32)
   
            hx = torch.zeros(1, env.batch_size, args['hidden_dim'])
            cx = torch.zeros(1, env.batch_size, args['hidden_dim'])
            last_hh = (hx, cx)
            logs = []
            actions = []
            probs = []
            terminated = np.zeros(env.batch_size).astype(np.float32)
   
            decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
            time_step = 0
            
            while time_step < args['decode_len']:
                dynamic = torch.from_numpy(state)
                terminated = torch.from_numpy(terminated)
                avail_actions = torch.from_numpy(avail_actions.reshape([env.batch_size, env.n_nodes]).astype(np.float32))
                idx, prob, logp, last_hh = actor.forward(static_hidden, static_ch_l, dynamic, decoder_input, last_hh, terminated, avail_actions)
                logs.append(logp.unsqueeze(1))
                actions.append(idx.unsqueeze(1))
                probs.append(prob.unsqueeze(1))
                state, avail_actions, reward, terminated, EV_vec, charge_vec, current_time = env.step(idx, EV_vec, charge_vec, current_time)
                decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                time_step += 1
            
            print("epochs: ", i)

            actions = torch.cat(actions, dim=1)  # (batch_size, seq_len)
            logs = torch.cat(logs, dim=1)  # (batch_size, seq_len)

            # Query the critic for an estimate of the reward
            critic_est = critic(torch.from_numpy(static), torch.from_numpy(ch_l), torch.from_numpy(dynamic_init)).view(-1)
            R = current_time.astype(np.float32)
            R = torch.from_numpy(R)
            advantage = (R - critic_est)
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
            print("e_t: ", e_t)
            if i % args['test_interval'] == 0:
        
                R = self.test(False)
                r_test.append(R)
                np.savetxt("trained_models/single/test_rewards.txt", r_test)
            
                print("testing average rewards: ", R)
                if R < best_model:
                    best_model = R
                    num = str(i // args['save_interval'])
                    torch.save(actor.state_dict(), 'trained_models/single' + '/' + 'best_model' + '_actor_params.pkl')
                    torch.save(critic.state_dict(), 'trained_models/single' + '/' + 'best_model' + '_critic_params.pkl')
                
                
        
            if i % args['save_interval'] ==0:
                num = str(i // args['save_interval'])
                torch.save(actor.state_dict(), 'trained_models/single' + '/' + num + '_actor_params.pkl')
                torch.save(critic.state_dict(), 'trained_models/single' + '/' + num + '_critic_params.pkl')

    def test(self, inference):
        args = self.args 
        env = self.env 
        dataGen = self.dataGen
        actor = self.actor 
        critic = self.critic 
        
        actor.eval()
        data = dataGen.get_test_all()
        env.input_data = data 
        state, avail_actions = env.reset()
        static = env.input_data[:, :, :2].reshape(env.batch_size, 2, env.n_nodes).astype(np.float32)
        ch_l = env.ch_l.reshape(env.batch_size, 1, env.n_nodes).astype(np.float32)
        EV_vec = []
        charge_vec = []
        for j in range(env.batch_size):
            EV_vec.append([])
            charge_vec.append([])
        current_time = np.zeros([env.batch_size])
        actions = []
        with torch.no_grad():
            static_hidden, static_ch_l = actor.emd_stat(torch.from_numpy(static), torch.from_numpy(ch_l))
            hx = torch.zeros(1, env.batch_size, args['hidden_dim'])
            cx = torch.zeros(1, env.batch_size, args['hidden_dim'])
            last_hh = (hx, cx)
            terminated = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
            time_step = 0
            
            while time_step < args['decode_len']:
     
                dynamic = torch.from_numpy(state)
                terminated = torch.from_numpy(terminated)
                avail_actions = torch.from_numpy(avail_actions)
                idx, prob, logp, last_hh = actor.forward(static_hidden, static_ch_l, dynamic, decoder_input, last_hh, terminated, avail_actions)
                state, avail_actions, reward, terminated, EV_vec, charge_vec, current_time = env.step(idx, EV_vec, charge_vec, current_time)
                decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                time_step += 1
                actions.append(idx)
            
        R = copy.copy(current_time)
        print("finished: ", sum(terminated))
        if inference:
            fname = 'test_results-{}-{}-len-{}-{}.txt'.format(args['test_size'], args['n_agents'], 
                                                               args['n_nodes'], args['difficulty'])
            fname = 'results/single/' + fname
            np.savetxt(fname, R)
        actor.train()
        return np.mean(R)