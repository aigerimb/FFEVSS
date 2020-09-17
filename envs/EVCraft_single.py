import numpy as np
import os
import warnings
import collections
import copy
import time 

def create_test_dataset(
                    args):
    
    rnd = np.random.RandomState(seed=args['random_seed'])
    n_problems = args['test_size']
    n_nodes = args['n_nodes']
    n_agents = args['n_agents']
    data_dir = args['data_dir']
    # build task name and datafiles
    task_name = 'FFEVSS-size-{}-{}-len-{}-{}.txt'.format(n_problems, n_agents, n_nodes, args['difficulty'])
    fname = os.path.join(data_dir, task_name)

    # create/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname,delimiter=' ')
        data = data.reshape(-1, n_nodes,4)
        input_data = data
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems
        input_pnt = np.random.uniform(0,1,
                                      size=(args['test_size'],args['n_nodes'],2))

        demand = np.ones([args['test_size'], args['n_demand']])
        charge_st = np.ones([args['test_size'], args['n_charge']])*2
        n = args['n_nodes']-1-args['n_demand']-args['n_charge']
        supply = np.ones([args['test_size'], n])*3

        network = np.concatenate([demand, charge_st, supply], 1)

        ch_l = np.zeros([args['test_size'], args['n_nodes']])
        for i in range(args['test_size']):
            np.random.shuffle(network[i])
            if args['difficulty'] == 'easy':
                k = 0
                for j in range(len(network[i])):
                    if network[i, j] == 3:
                        if k < n/2:
                            ch_l[i, j] = np.random.randint(4, 6)
                            k += 1
                        else:
                            ch_l[i, j] = np.random.randint(1, 4)
                            k += 1
            else:
                for j in range(len(network[i])):
                    if network[i, j] == 3:
                        ch_l[i, j] = np.random.randint(1, 4)

        network = np.concatenate([network, np.zeros([args['test_size'], 1])], 1)
        input_data = np.concatenate([input_pnt, np.expand_dims(network, 2), np.expand_dims(ch_l, 2)], 2)

        np.savetxt(fname, input_data.reshape(-1, n_nodes*4))

    return input_data

class DataGenerator(object):
    def __init__(self,
                 args):

        '''
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['batch_size']: batchsize for training

        '''
        self.args = args
        self.rnd = np.random.RandomState(seed= args['random_seed'])


        # create test data
        self.test_data = create_test_dataset(args)


        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x 3]
        '''
        args = self.args
        input_pnt = np.random.uniform(0,1,
                                      size=(args['batch_size'],args['n_nodes'],2))

        demand = np.ones([args['batch_size'], args['n_demand']])
        charge_st = np.ones([args['batch_size'], args['n_charge']])*2
        n = args['n_nodes']-1-args['n_demand']-args['n_charge']
        supply = np.ones([args['batch_size'], n])*3

        network = np.concatenate([demand, charge_st, supply], 1)

        ch_l = np.zeros([args['batch_size'], args['n_nodes']])
        for i in range(args['batch_size']):
            np.random.shuffle(network[i])
            if args['difficulty'] == 'easy':
                k = 0
                for j in range(len(network[i])):
                    if network[i, j] == 3:
                        if k < n/2:
                            ch_l[i, j] = np.random.randint(4, 6)
                            k += 1
                        else:
                            ch_l[i, j] = np.random.randint(1, 4)
                            k += 1
            else:
                for j in range(len(network[i])):
                    if network[i, j] == 3:
                        ch_l[i, j] = np.random.randint(1, 4)

        network = np.concatenate([network, np.zeros([args['batch_size'], 1])], 1)
        input_data = np.concatenate([input_pnt, np.expand_dims(network, 2), np.expand_dims(ch_l, 2)], 2)

        return input_data


    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        if self.count<self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count+self.args['test_size']]
            self.count +=self.args['test_size']
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count+self.args['test_size']]
            self.count +=self.args['test_size']

        return input_pnt

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data
    

    
    
class Env(object):
    def __init__(self, args, data):
       
        self.args = args
        self.rnd = np.random.RandomState(seed= args['random_seed'])
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        
        self.n_drivers = args['n_drivers']
        self.input_dim = args['input_dim']
        self.input_data = data

        self.batch_size = args['batch_size']
        self.load = np.ones([1])*self.n_drivers
     
        self.n_charge = args['n_charge']
        self.n_demand = args['n_demand']
       
        
       
        
    def reset(self):
    
        
        self.load = np.ones([self.batch_size])*self.n_drivers
        self.input_pnt = self.input_data[:, :, :2]
        self.network = self.input_data[:, :, 2]
        self.ch_l = self.input_data[:, :, 3]
        self.shuttle_loc = np.ones([self.batch_size])*(self.n_nodes -1)
 
        self.d = (np.ones([self.batch_size, self.n_nodes])==self.network).astype(int)
        self.s =(np.ones([self.batch_size, self.n_nodes])*3==self.network).astype(int)
        self.ch = (np.ones([self.batch_size, self.n_nodes])*2==self.network).astype(int)
        self.avail_actions =  np.reshape(self.s, [self.batch_size, self.n_nodes])
       
        # create distance matrix
        self.dist_mat = np.zeros([self.batch_size, self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                self.dist_mat[:, i, j] = ((self.input_pnt[:, i, 0] - self.input_pnt[:, j, 0])**2 + (self.input_pnt[:, i, 1] - self.input_pnt[:, j, 1])**2)**0.5
                self.dist_mat[:, j, i] =  self.dist_mat[:, i, j]
                
        self.time_mat = self.dist_mat*4/3
        self.ch_time = np.reshape(self.time_mat[np.nonzero(self.time_mat)], [self.batch_size, self.n_nodes, self.n_nodes-1])
        self.ch_time = np.mean(self.ch_time, axis = (1, 2))
        self.dem_fill = copy.copy(self.time_mat)
        self.evs = []
        
        for k in range(self.batch_size):
            self.evs.append([])
            for i in range(self.n_nodes):
                 if self.network[k, i] != 1:
                    self.dem_fill[k, :, i] = np.ones(self.n_nodes)*10000
                    self.dem_fill[k, i, i] = 10000
                 
        
        self.rad = self.ch_time*3/4
        
        self.global_state = np.zeros([self.batch_size, self.n_nodes, 6])
        self.global_state[:, :, 0] = self.d*(-1) + self.s
        self.global_state[:, :, 1] = self.ch_l
        self.global_state[:, self.n_nodes-1, 5] = self.load 
        
     #   e_t = time.time() - s_t
     #   print("reset time: ", e_t)
        s =  np.zeros([self.batch_size, self.n_nodes, 3])
        s[:, :, 0] = np.greater(self.global_state[:, :, 0], 0).astype(int) 
        s[:, :, 1] = -np.greater(self.global_state[:, :, 0], 0).astype(int) + np.expand_dims(self.load, 1) 
        s[:, :, 2] = self.dist_mat[:, :, self.n_nodes-1]
        s = s.astype(np.float32)
        self.avail_actions = self.avail_actions.astype(np.float32)
        return s, self.avail_actions
    
    
    def get_avail_agent_actions(self, agent_id):
        return self.avail_actions
    
    def get_rewards(self):
        return self.final_rewards
    
       
    def step(self, idx, EV_vec, charge_vec, current_time):
    #    s_t = time.time()
   
        idx = np.array(idx)
        
        
        # determine the time step 
        time_step = np.zeros([self.batch_size])
        for k in range(self.batch_size):
            if idx[k] != self.shuttle_loc[k]:
                time_step[k] = self.time_mat[k, int(self.shuttle_loc[k]), int(idx[k])]
            else:
                t_min = 10000
                ev =-1
                for i in range(len(EV_vec[k])):
                    if EV_vec[k][i][0] == idx[k]:
                        t_min = EV_vec[k][i][1]
                        ev= i
                tt_min = 10000
                chv =-1
                for i in range(len(charge_vec[k])):
                    if charge_vec[k][i][0] == idx[k]:
                        tt_min = charge_vec[k][i][1]
                        chv= i
                if (ev+chv) >-2:
                    if t_min > tt_min:
                        time_step[k] = tt_min 
                    else:
                        time_step[k] = t_min
     #   e_t = s_t - time.time()
     #   print("time step calculation: ", e_t)
    #    s_t = time.time()
        # update charge_vec and EV_vec according to time step
        EV_vec_new = []
        charge_vec_new = []
        idx_dd = []
        idx_ch = []
        for k in range(self.batch_size):
            EV_vec_new.append([])
            charge_vec_new.append([])
            idx_dd.append([])
            idx_ch.append([])
            for j in range(len(EV_vec[k])):
                n = EV_vec[k][j][0]
                if EV_vec[k][j][1] > time_step[k]:
                    EV_vec[k][j][1] -= time_step[k]
                    EV_vec_new[k].append(EV_vec[k][j])
                   
                    # realese transitions
                else:
                    # fully realse
                    if EV_vec[k][j][2] > 3:
                        self.global_state[k, n, 0] +=1
                        self.global_state[k, n, 1] += EV_vec[k][j][2]
                        self.global_state[k, n, 2] += 1
                        self.global_state[k, n, 3] = 0
                        # release driver add to charge_vec
                    else:
                        EV_vec[k][j][1] = (5-EV_vec[k][j][2])*self.ch_time[k] + (EV_vec[k][j][1] - time_step[k])
                        if EV_vec[k][j][1] > 0:
                            self.global_state[k, n, 2] += 1
                            self.global_state[k, n, 3] = 0
                            self.global_state[k, n, 4] = 1
                            charge_vec_new[k].append(EV_vec[k][j])
                        else:
                            self.global_state[k, n, 0] +=1
                            self.global_state[k, n, 1] += 5
                            self.global_state[k, n, 2] += 1
                            self.global_state[k, n, 3] = 0
                            
            for j in range(len(charge_vec[k])):
                if charge_vec[k][j][1] > time_step[k]:
                    v = charge_vec[k][j][1] - time_step[k]
                    n = int(charge_vec[k][j][0])
                    charge_vec_new[k].append([charge_vec[k][j][0], v, charge_vec[k][j][2]])
                    
                    # release transitions from charge_vec 
                else:
                    n = charge_vec[k][j][0]
                    self.global_state[k, n, 0] += 1
                    self.global_state[k, n, 1] += 5
                    self.global_state[k, n, 4] = 0
                    
            for j in range(self.n_nodes):
                if self.global_state[k, j, 0] > 0 and self.global_state[k, j, 2] > 0:
                    if self.global_state[k, j, 1] > 3:
                        idx_dd[k].append([j, self.global_state[k, j, 1]])
                    else:
                        idx_ch[k].append([j, self.global_state[k, j, 1]])
                    self.global_state[k, j, 0] -=1
                    self.global_state[k, j, 1] = 0
                    self.global_state[k, j, 2] -= 1
                    
        EV_vec = EV_vec_new
        charge_vec = charge_vec_new
        # check if there are available driver and EV to go to places
       
     #   e_t = s_t - time.time()
     #   print("EV and charge vecs update: ", e_t)
                    
         ##################################################################################
                              # movements of shuttles
        #################################################################################
     #   s_t = time.time()
        for k in range(self.batch_size):
            n = int(idx[k])
            # if selected node is a supplier node 
            if self.network[k, n] == 3 and self.global_state[k, n, 0] > 0:
                if np.sum(np.less(self.global_state[k, :, 0], 0).astype(int)) > 0:
                    # if there is a driver to drop off
                    if self.load[k] > 0:
                        self.global_state[k, n, 0] -= 1
                        self.load[k] -=1
                        if self.global_state[k, n, 1] > 3:
                            idx_dd[k].append([n, copy.copy(self.global_state[k, n, 1])])
                        else:
                            idx_ch[k].append([n, copy.copy(self.global_state[k, n, 1])])
                        self.global_state[k, n, 1] = 0
            # if the selected node is demand node and there is a driver to pick up
            elif self.network[k, n] == 1 and self.global_state[k, n, 2] > 0:
                # if there is capacity to fit driver
                if self.capacity - self.load[k] > 0:
                    self.global_state[k, n, 2] -= 1
                    self.load[k] +=1
                # if the selected node is charging node
            elif self.network[k, n] == 2 :
                # if there is a driver and a capacity to fit driver
                if self.global_state[k, n, 2] > 0:
                    if self.capacity - self.load[k] > 0:
                        self.global_state[k, n, 2] -= 1
                        self.load[k] +=1
                # if there is ev and there is a driver to drop off
                if self.global_state[k, n, 0] > 0:
                    if self.load[k] > 0:
                        self.global_state[k, n, 0] -= 1
                        self.load[k] -=1
                        if self.global_state[k, n, 1] > 3:
                            idx_dd[k].append([n, copy.deepcopy(self.global_state[k, n, 1])])
                        else:
                            idx_ch[k].append([n, copy.deepcopy(self.global_state[k, n, 1])])
                        self.global_state[k, n, 1] = 0
            

    #    e_t = s_t - time.time()
     #   print("shuttle movement: ", e_t)
        ##################################################################################
                              # determine delayed movements of EVs
        #################################################################################
        # update current dem_fill matrix: for demand 0 nodes and expected transitions
        # obtain vector of charging stations status
     #   s_t = time.time()
      


        # for EVs directly heading to the demand nodes determine the closest unfilled demand node
        for l in range(self.batch_size):
            for k in range(len(idx_dd[l])):
                j = int(idx_dd[l][k][0])
                n = np.argmin(self.dem_fill[l, j, :])
                if self.global_state[l, n, 0] < 0:
                    d = self.time_mat[l, j, n]
                    EV_vec[l].append([n, d, idx_dd[l][k][1]])
                    self.global_state[l, n, 3] = 1
                    self.dem_fill[l, :, n] = 10000*np.ones(self.n_nodes)
                    self.evs[l].append([j, n])
                    # if there is no expected unfulfilled demand 
                else:
                    self.global_state[l, j, 0] +=1
                    self.global_state[l, j, 1] += idx_dd[l][k][1]
                    if idx[l] == j:
                        self.load[l] += 1
                    else:
                        self.global_state[l, j, 2] += 1
                        
            ch_station = (np.equal(self.global_state[l, :, 4]+ self.global_state[l, :, 3]+self.global_state[l, :, 0], 0).astype(int))*self.ch[l]
          
          #  print(ch_station)
            for k in range(len(idx_ch[l])):
                j = int(idx_ch[l][k][0])
                nn = ch_station*self.time_mat[l, j, :]
                if sum(nn) > 0:
                    for p in range(len(nn)):
                        if nn[p] == 0:
                            nn[p] = 10000
                    m = np.argmin(nn)
                    EV_vec[l].append([m, nn[int(m)], idx_ch[l][k][1]]) 
                    self.global_state[l, m, 3] = 1
                    ch_station[m] = 0
                    self.evs[l].append([j, m])
                    # if there is no free charging nodes
                else:
                    self.global_state[l, j, 0] +=1
                    self.global_state[l, j, 1] = idx_ch[l][k][1]
                    if idx[l] == j :
                        self.load[l] +=1
                    else:
                        self.global_state[l, j, 2] += 1
            self.global_state[l, int(self.shuttle_loc[l]), 5] = 0
            self.global_state[l, int(idx[l]), 5] = self.load[l]
            self.shuttle_loc[l] = idx[l]
                        
    #    e_t = s_t - time.time()
    #    print("EV relocation: ", e_t)
    #    s_t = time.time()
  
        current_time += time_step 
       
        self.avail_actions = np.zeros([self.batch_size, self.n_nodes])
        md = np.sum(np.less(self.global_state[:, :, 0], 0), 1)
        ml = np.sum(np.greater(self.global_state[:, :, 2], 0), 1)
        mask = np.zeros([self.batch_size, self.n_nodes])
        terminated = np.zeros(self.batch_size)
       
        # mask nodes without a driver or expected driver if load = 0 or all demand is fulfilled 
        for k in range(self.batch_size):
            if self.load[k] == 0 or np.sum(np.less(self.dem_fill[k], 10000)) == 0:
                mask[k] = (np.greater(self.global_state[k, :, 2], 0)).astype(int) + (np.greater(self.global_state[k, :, 3], 0)).astype(int)
                # if there is a driver and we still have unsatisfied demand 
            else:

                # mask all demand nodes if there is no driver and no expected driver
                # mask all supply nodes if there is no ev or driver
                # mask charging nodes if there is no driver or expected driver or EV
                mdr = self.d[k]*self.global_state[k, :, 2] + self.d[k]*self.global_state[k, :, 3]
                ms = self.s[k]*self.global_state[k, :, 0] + self.s[k]*self.global_state[k, :, 2]
                mch = self.ch[k]*self.global_state[k, :, 2] + self.ch[k]*self.global_state[k, :, 3] + self.ch[k]*self.global_state[k, :, 0] + self.ch[k]*self.global_state[k, :, 4]

                mask[k] = np.concatenate([np.greater(mdr+ms+mch, 0).astype(int)[:-1],
                                                      np.zeros([1])],0)
            
                # mask supplier nodes if all charging nodes are either have EVs or charging 
                n_ch = (np.greater(self.global_state[k, :, 0], 0).astype(int) + np.greater(self.global_state[k, :, 4], 0).astype(int) + np.greater(self.global_state[k, :, 3], 0).astype(int))*self.ch[k]
                if np.sum(n_ch) == self.n_charge:
                    mask[k] -= self.s[k]
          
                # convert mask to new setting
            #    mask[k] = np.equal(mask[k],0).astype(int)
          #      if self.ch[k, int(self.shuttle_loc[k])] == 1 and self.global_state[k, int(self.shuttle_loc[k]), 4] ==1:
          #          mask[k, int(self.shuttle_loc[k])] = 0
              
            mask[k] = np.greater(mask[k], 0).astype(int)
   #         print(mask[k])
            self.avail_actions[k] =  mask[k]
            
        #    self.avail_actions[k] = np.less_equal(self.dist_mat[k, int(self.shuttle_loc[k]), :], np.ones([self.n_nodes])*self.rad[k])*mask[k] 

            if md[k] == 0 and ml[k] == 0 and self.shuttle_loc[k] == self.n_nodes-1:
                self.avail_actions[k, self.n_nodes-1] = 1
                terminated[k] = 1
            elif md[k] == 0 and ml[k] == 0:
                self.avail_actions[k, self.n_nodes-1] = 1
            else:
                self.avail_actions[k, self.n_nodes-1] = 0
            if sum(self.avail_actions[k]) == 0:
             #   aa = mask[k]*self.dist_mat[k, int(self.shuttle_loc[k]),:]
             #   aa[aa==0] = 1000
             #   g = np.argmin(aa)
             #   self.avail_actions[k,  g] = 1
                print("avail_actions zero, smth is wrong")
                  
    
        reward = -(time_step)
        s =  np.zeros([self.batch_size, self.n_nodes, 3])
        s[:, :, 0] = np.greater(self.global_state[:, :, 0], 0).astype(int) + self.global_state[:, :, 2] + self.global_state[:, :, 3] + self.global_state[:, :, 4] 
        s[:, :, 1] = -np.greater(self.global_state[:, :, 0], 0).astype(int) + self.global_state[:, :, 2] + self.global_state[:, :, 3] - self.global_state[:, :, 4] + np.expand_dims(self.load, 1)
        s[:, :, 2] = self.dist_mat[np.arange(self.batch_size, dtype=int), self.shuttle_loc.astype(int)]
        s = s.astype(np.float32)
        self.avail_actions = self.avail_actions.astype(np.float32)
        
        self.final_rewards = current_time 
        terminated = terminated.astype(np.float32)
   #     print("avai: ", self.avail_actions[127])
   #     print(self.global_state[127])                      
        return s, self.avail_actions, reward, terminated, EV_vec, charge_vec, current_time