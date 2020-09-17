import numpy as np
import os
import warnings
import collections
import copy

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

    # cteate/load data
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

# global state dim: [n_nodes, 8]
# 0 - x locs
# 1-y locs
# 2 - # of Evs, if EV needed negative
# 3 - ch_l of EV
# 4 - # of drivers
# 5 - expected transition
# 6 - # of shuttles located
# 7 - total load


# local state dime [batch_size, n_agents, None, 6]
#  0 - # node ID
# 1 - # of Evs, if EV needed negative
# 2 - ch_l of EV
# 3 - # of drivers
# 4 - # of shuttles located x by load of each shuttle
# 5 - if there is expected transition


# currently have mask for global state
# need to mask based on load for each ev
# did not mask nodes if it is in charging mode


class Env(object):
    def __init__(self, args, data):
        '''
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 2
        '''
        self.args = args

        self.rnd = np.random.RandomState(seed= args['random_seed'])
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_drivers = args['n_drivers']
        self.input_dim = args['input_dim']
        self.n_agents = args['n_agents']
        self.input_data = data

        self.batch_size = args['batch_size']
     #   self.batch_size = 2
        self.state_shape = self.n_nodes*9
        self.obs_shape = self.n_nodes*8
        self.n_charge = args['n_charge']
        self.n_demand = args['n_demand']
        


    def reset(self):
        
        self.load = np.ones([self.batch_size, self.n_agents])*self.n_drivers
        self.input_pnt = self.input_data[:, :, :2]
        self.network = self.input_data[:, :, 2]
        self.ch_l = self.input_data[:, :, 3]
        self.shuttle_loc = np.ones([self.batch_size, self.n_agents])*(self.n_nodes -1)
 
        self.d = (np.ones([self.batch_size, self.n_nodes])==self.network).astype(int)
        self.s =(np.ones([self.batch_size, self.n_nodes])*3==self.network).astype(int)
        self.ch = (np.ones([self.batch_size, self.n_nodes])*2==self.network).astype(int)
       
        # create distance matrix
        self.dist_mat = np.zeros([self.batch_size, self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                self.dist_mat[:, i, j] = ((self.input_pnt[:, i, 0] - self.input_pnt[:, j, 0])**2 + (self.input_pnt[:, i, 1] - self.input_pnt[:, j, 1])**2)**0.5
                self.dist_mat[:, j, i] =  self.dist_mat[:, i, j]
                
        self.time_mat = self.dist_mat*4/3
        self.ch_time = np.reshape(self.time_mat[np.nonzero(self.time_mat)], [self.batch_size, self.n_nodes, self.n_nodes-1])
        self.rad = np.quantile(self.ch_time*3/4, 1, axis=(1, 2))
        self.ch_time = np.mean(self.ch_time, axis = (1, 2))
        self.dem_fill = copy.copy(self.time_mat)
        print("ch time: ", self.ch_time[0])
        
        for k in range(self.batch_size):
            for i in range(self.n_nodes):
                 if self.network[k, i] != 1:
                    self.dem_fill[k, :, i] = np.ones(self.n_nodes)*10000
                    self.dem_fill[k, i, i] = 10000
                 
        
   #     self.rad = self.ch_time*3/4
        self.global_state = np.zeros([self.batch_size, self.n_nodes, 6])
        self.global_state[:, :, 0] = self.d*(-1) + self.s
        self.global_state[:, :, 1] = self.ch_l
        self.global_state[:, self.n_nodes-1, 5] = self.n_drivers*self.n_agents 
        
        s =  np.zeros([self.batch_size, self.n_nodes, 1])
        s[:, :, 0] = np.greater(self.global_state[:, :, 0], 0).astype(int) 
        
        s = s.astype(np.float32)
        stat_coord = np.zeros([self.batch_size, self.n_agents, self.n_nodes, 2])
        stat_ch_l = np.zeros([self.batch_size, self.n_agents, self.n_nodes, 1])
        obs = np.zeros([self.batch_size, self.n_agents, self.n_nodes, 3])
        self.avail_actions = np.zeros([self.batch_size, self.n_agents, self.n_nodes])
        self.dones = []
        self.evs = []
        for k in range(self.batch_size):
            self.dones.append([])
            self.evs.append([])
            s1 = -np.greater(self.global_state[:, :, 0], 0).astype(int) 
            s2 = self.dist_mat[:, :, self.n_nodes-1]
        #    A = self.dist_mat[k, self.n_nodes-1]*self.s[k]
       #     A[A==0] =10000
            # find closest supply nodes
       #     actions = np.argsort(A)[:self.n_agents]
        #    for n in actions:
        #        self.avail_actions[k, :, n] = 1 
            for a in range(self.n_agents):
                self.avail_actions[k, a, :] = self.s[k] 
                t = np.less_equal(self.dist_mat[k, self.n_nodes-1, :], np.ones([self.n_nodes])*self.rad[k])
                obs[k, a, :, 0] = t*s[k, :, 0]
                obs[k, a, :, 1] = t*(s1[k, :] + self.load[k, a])
                obs[k, a, :, 2] = t*s2[k, :]
                stat_coord[k, a, :, 0] = t*self.input_pnt[k, :, 0]
                stat_coord[k, a, :, 1] = t*self.input_pnt[k, :, 1]
                stat_ch_l[k, a, :, 0] = t*self.ch_l[k, :]
            
        obs = obs.astype(np.float32)
        stat_coord = stat_coord.astype(np.float32)
        stat_ch_l = stat_ch_l.astype(np.float32)
        return s, obs, self.avail_actions, stat_coord, stat_ch_l 


    def step(self, idx, EV_vec, charge_vec, current_time, terminated, sel_a, sh_vec):
      #  print("idx: ", idx)
        old_load = copy.copy(self.load)
       
        time_step = np.zeros(self.batch_size)
        times = np.zeros([self.batch_size, self.n_agents])
        baseline = np.zeros([self.batch_size, self.n_agents])
        new_sel_a = []
        new_sh_vec = []
        for k in range(self.batch_size):
            new_sel_a.append([])
            new_sh_vec.append([])
            if terminated[k] != 1:
                tr_time = np.zeros([self.n_agents])
                new_sel_a_k = []
                if len(sel_a[k]) == 0:
                    new_sel_a_k = np.arange(self.n_agents)
                else:
                    for a in sel_a[k]:
                        if self.shuttle_loc[k, a] == self.n_nodes-1 and idx[k, a] == self.n_nodes-1:
                            self.dones[k].append(a)
                        else:
                            new_sel_a_k.append(a)
                sel_a[k] = copy.copy(new_sel_a_k) 
                actions = 1000
                for a in sel_a[k]:
                    time = self.time_mat[k, int(self.shuttle_loc[k, a]), int(idx[k, a])]
                    times[k, a] = time
                    baseline[k, a] = time 
                    tr_time[a] = time
                    if time < actions:
                        actions = time
           
                    if times[k, a] == 0:
                        t_min = 1000
                        for i in range(len(EV_vec[k])):
                            if EV_vec[k][i][0] == idx[k, a] :
                                if EV_vec[k][i][1] < t_min:
                                    t_min = EV_vec[k][i][1]
                            
                        tt_min = 1000
                        for i in range(len(charge_vec[k])):
                            if charge_vec[k][i][0] == idx[k, a]:
                                if charge_vec[k][i][1] < tt_min:
                                    tt_min = charge_vec[k][i][1]
                        
                        if t_min > tt_min:
                            actions = tt_min  
                            tr_time[a] = tt_min
                            baseline[k, a] = tt_min
                        else:
                            actions = t_min 
                            tr_time[a] = t_min
                            baseline[k, a] = t_min
                for j in range(len(sh_vec[k])):
                    if sh_vec[k][j][1] < actions:
                        actions = sh_vec[k][j][1]
                    
            
                time_step[k] = actions 
           
                for a in sel_a[k]:
                    if tr_time[a] > time_step[k]:
                        new_sh_vec[k].append((idx[k, a], tr_time[a]-time_step[k], a))
                    else:
                        new_sel_a[k].append(a)
            
                for j in range(len(sh_vec[k])):
                    if sh_vec[k][j][1] > time_step[k]:
                        new_sh_vec[k].append((sh_vec[k][j][0], sh_vec[k][j][1] - time_step[k], sh_vec[k][j][2]))
                    else:
                        new_sel_a[k].append(sh_vec[k][j][2])
            
                    
        sel_a = new_sel_a
        sh_vec = new_sh_vec 
      
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
                    if np.sum(np.less(self.dem_fill[k], 10000)) > 0:
                        if self.global_state[k, j, 1] > 3:
                            idx_dd[k].append([j, self.global_state[k, j, 1]])
                        else:
                            idx_ch[k].append([j, self.global_state[k, j, 1]])
                        self.global_state[k, j, 0] -=1
                        self.global_state[k, j, 1] = 0
                        self.global_state[k, j, 2] -= 1
                    
        EV_vec = EV_vec_new
        charge_vec = charge_vec_new              
          
        
        ##################################################################################
                              # movements of shuttles
        #################################################################################
        for k in range(self.batch_size):
            for a in sel_a[k]:
                n = int(idx[k, a])
                # if selected node is a supplier node 
                if self.network[k, n] == 3 and self.global_state[k, n, 0] > 0:
                    if np.sum(np.less(self.dem_fill[k], 10000)) > 0:
                        # if there is a driver to drop off
                        if self.load[k, a] > 0:
                            self.global_state[k, n, 0] -= 1
                            self.load[k, a] -=1
                            if self.global_state[k, n, 1] > 3:
                                idx_dd[k].append([n, copy.copy(self.global_state[k, n, 1]), a])
                            else:
                                idx_ch[k].append([n, copy.copy(self.global_state[k, n, 1]), a])
                            self.global_state[k, n, 1] = 0
                # if the selected node is demand node and there is a driver to pick up
                elif self.network[k, n] == 1 and self.global_state[k, n, 2] > 0:
                    # if there is capacity to fit driver
                    if self.capacity - self.load[k, a] > 0:
                        self.global_state[k, n, 2] -= 1
                        self.load[k, a] +=1
                # if the selected node is charging node
                elif self.network[k, n] == 2 :
                    # if there is a driver and a capacity to fit driver
                    if self.global_state[k, n, 2] > 0:
                        if self.capacity - self.load[k, a] > 0:
                            self.global_state[k, n, 2] -= 1
                            self.load[k, a] +=1
                    # if there is ev and there is a driver to drop off
                    if self.global_state[k, n, 0] > 0 and np.sum(np.less(self.dem_fill[k], 10000)) > 0:
                        if self.load[k, a] > 0:
                            self.global_state[k, n, 0] -= 1
                            self.load[k, a] -=1
                            if self.global_state[k, n, 1] > 3:
                                idx_dd[k].append([n, copy.deepcopy(self.global_state[k, n, 1]), a])
                            else:
                                idx_ch[k].append([n, copy.deepcopy(self.global_state[k, n, 1]), a])
                            self.global_state[k, n, 1] = 0
            
        ##################################################################################
                              # determine delayed movements of EVs
        #################################################################################
       
        # for EVs directly heading to the demand nodes determine the closest unfilled demand node
        for l in range(self.batch_size):
            for k in range(len(idx_dd[l])):
                j = int(idx_dd[l][k][0])
                n = np.argmin(self.dem_fill[l, j, :])
                if self.global_state[l, n, 0] < 0 and self.global_state[l, n, 3]==0:
                    if len(idx_dd[l][k]) > 2: 
                        a = idx_dd[l][k][2]
                        d = self.time_mat[l, j, n] 
                        if d > 0:
                            EV_vec[l].append([n, d, idx_dd[l][k][1]])
                            self.global_state[l, n, 3] = 1
                        else:
                            self.global_state[l, n, 0] +=1
                            self.global_state[l, n, 2] +=1
                    else:
                         d = self.time_mat[l, j, n]
                         EV_vec[l].append([n, d, idx_dd[l][k][1]])
                         self.global_state[l, n, 3] = 1
                    self.dem_fill[l, :, n] = 10000*np.ones(self.n_nodes)
                    # if there is no expected unfulfilled demand
                    self.evs[l].append([j, n])
                else:
                    self.global_state[l, j, 0] +=1
                    self.global_state[l, j, 1] += idx_dd[l][k][1]
                   
                    if len(idx_dd[l][k]) > 2: 
                        a = idx_dd[l][k][2]
                        self.load[l, a] += 1
                    else:
                        self.global_state[l, j, 2] += 1
                    
                        
            ch_station = (np.equal(self.global_state[l, :, 4]+ self.global_state[l, :, 3]+self.global_state[l, :, 0], 0).astype(int))*self.ch[l]
            for k in range(len(idx_ch[l])):
                j = int(idx_ch[l][k][0])
                nn = ch_station*self.time_mat[l, j, :]
                if sum(nn) > 0:
                    for p in range(len(nn)):
                        if nn[p] == 0:
                            nn[p] = 10000
                    m = np.argmin(nn)
                    if len(idx_ch[l][k]) > 2:
                        a = idx_ch[l][k][2]
                        d = nn[int(m)] 
                        if d > 0: 
                            EV_vec[l].append([m, d, idx_ch[l][k][1]]) 
                            self.global_state[l, m, 3] = 1
                            ch_station[m] = 0
                            self.evs[l].append([j, m])
                        else:
                            d = (5-idx_ch[l][k][1])*self.ch_time[l] + d
                            if d > 0:
                                charge_vec[l].append([m, d, idx_ch[l][k][1]]) 
                                self.global_state[l, m, 4] = 1
                            else:
                                self.global_state[l, m, 0] = +1
                                self.global_state[l, m, 1] = 5
                            self.global_state[l, m, 2] = +1
                            ch_station[m] = 0
                           
                    # if there is no free charging nodes
                else:
                    self.global_state[l, j, 0] +=1
                    self.global_state[l, j, 1] = idx_ch[l][k][1]
                    if len(idx_ch[l][k]) > 2:
                        a = idx_ch[l][k][2]
                        self.load[l, a] += 1
                    else:
                        self.global_state[l, j, 2] += 1
                   
            for a in sel_a[l]:
                self.global_state[l, int(self.shuttle_loc[l, a]), 5] -= old_load[l, a]
                self.global_state[l, int(idx[l, a]), 5] += self.load[l, a]
                self.shuttle_loc[l, a] = idx[l, a]
                if self.global_state[l, int(idx[l, a]), 2] > 0:
                    self.load[l, a] += 1
                    self.global_state[l, idx[l, a], 2] -= 1

        current_time += time_step 
        self.avail_actions = np.zeros([self.batch_size, self.n_agents, self.n_nodes])
        md = np.sum(np.less(self.global_state[:, :, 0], 0), 1)
        ml = np.sum(np.greater(self.global_state[:, :, 2], 0), 1)
        mask = np.zeros([self.batch_size, self.n_agents, self.n_nodes])
        
        padded = np.zeros(self.batch_size)
        # mask nodes without a driver or expected driver if load = 0 or all demand is fulfilled 
        for k in range(self.batch_size):
            for a in sel_a[k]:
             #  print("r_dem: ", np.sum(np.less(self.dem_fill[k], 10000)))
                expected_dr = (np.greater(self.global_state[k, :, 2], 0)).astype(int) + (np.greater(self.global_state[k, :, 3], 0)).astype(int)
                if self.load[k, a] == 0 or np.sum(np.less(self.dem_fill[k], 10000)) == 0:
                    mask[k, a] = expected_dr
                # if there is a driver and we still have unsatisfied demand 
                
                else:
                    
                    # mask all demand nodes if there is no driver and no expected driver
                    # mask all supply nodes if there is no ev or driver
                    # mask charging nodes if there is no driver or expected driver or EV
                    mdr = self.d[k]*self.global_state[k, :, 2] + self.d[k]*self.global_state[k, :, 3]
                    ms = self.s[k]*self.global_state[k, :, 0] + self.s[k]*self.global_state[k, :, 2]
                    mch = self.ch[k]*self.global_state[k, :, 2] + self.ch[k]*self.global_state[k, :, 3] + self.ch[k]*self.global_state[k, :, 0] + self.ch[k]*self.global_state[k, :, 4]

                    mask[k, a] = np.concatenate([np.greater(mdr+ms+mch, 0).astype(int)[:-1],
                                                          np.zeros([1])],0)
                    
                    
                    # mask supplier nodes if all charging nodes are either have EVs or charging 
                    n_ch = (np.greater(self.global_state[k, :, 0], 0).astype(int) + np.greater(self.global_state[k, :, 4], 0).astype(int) + np.greater(self.global_state[k, :, 3], 0).astype(int))*self.ch[k]
                    if np.sum(n_ch) == self.n_charge:
                        mask[k, a] -= self.s[k]
                pots = np.zeros([self.n_nodes])
                for j in range(len(sh_vec[k])):
                    n = int(sh_vec[k][j][0])
                    pots[n] =1
                    mask[k, a, n] = 0
          
                mask[k, a] = np.greater(mask[k, a], 0).astype(int) 
           #     print("mask: ", mask[k, a])
                self.avail_actions[k, a] =  mask[k, a]
                # obtain local view 
                self.avail_actions[k, a] = np.less_equal(self.dist_mat[k, int(self.shuttle_loc[k, a]), :], np.ones([self.n_nodes])*self.rad[k])*mask[k, a] 
                
                if md[k] == 0 and ml[k] == 0 and self.shuttle_loc[k, a] == self.n_nodes-1:
                    self.avail_actions[k, a, self.n_nodes-1] = 1
                elif md[k] == 0 and ml[k] == 0:
                    self.avail_actions[k, a, self.n_nodes-1] = 1
                elif np.sum(np.less(self.dem_fill[k], 10000)) == 0 and np.sum(np.less(pots-expected_dr, 0)) == 0:
                    self.avail_actions[k, a, self.n_nodes-1] = 1
                else:
                    self.avail_actions[k, a, self.n_nodes-1] = 0
                if sum(self.avail_actions[k, a]) == 0:
        #            aa = mask[k, a]*self.dist_mat[k, int(self.shuttle_loc[k, a]),:]
        #            aa[aa==0] = 1000
        #            g = np.argmin(aa)
                    self.avail_actions[k, a, self.n_nodes-1] = 1
                    
            for j in range(len(sh_vec[k])):
                a = int(sh_vec[k][j][2])
                n = int(sh_vec[k][j][0])
                self.avail_actions[k, a, n] = 1
                
            for a in self.dones[k]:
                self.avail_actions[k, a, self.n_nodes-1] = 1
                   
            if md[k] == 0 and ml[k] == 0:
                if self.global_state[k, self.n_nodes-1, 5] == self.n_drivers*self.n_agents:
                    if terminated[k] == 0:
                        terminated[k] = 1
                    else:
                        padded[k] = 1
        
        reward = -(time_step).astype(np.float32)
        s =  np.zeros([self.batch_size, self.n_nodes, 1])
        s[:, :, 0] = np.greater(self.global_state[:, :, 0], 0).astype(int) + self.global_state[:, :, 2] + self.global_state[:, :, 3] + self.global_state[:, :, 4] 
       
        s = s.astype(np.float32)
        self.avail_actions = self.avail_actions.astype(np.float32)
        terminated = terminated.astype(np.float32)
        padded = padded.astype(np.float32)
        baseline = baseline.astype(np.float32)
        s =  np.zeros([self.batch_size, self.n_nodes, 1])
        s[:, :, 0] = np.greater(self.global_state[:, :, 0], 0).astype(int) + self.global_state[:, :, 2] + self.global_state[:, :, 3] + self.global_state[:, :, 4] 
        
        s = s.astype(np.float32)
        stat_coord = np.zeros([self.batch_size, self.n_agents, self.n_nodes, 2])
        stat_ch_l = np.zeros([self.batch_size, self.n_agents, self.n_nodes, 1])
        obs = np.zeros([self.batch_size, self.n_agents, self.n_nodes, 3])
        for k in range(self.batch_size):
            s1 = -np.greater(self.global_state[:, :, 0], 0).astype(int) + self.global_state[:, :, 2] + self.global_state[:, :, 3] - self.global_state[:, :, 4] 
            for a in range(self.n_agents):
                t = np.less_equal(self.dist_mat[k, int(self.shuttle_loc[k,a]), :], np.ones([self.n_nodes])*self.rad[k])
                obs[k, a, :, 0] = t*s[k, :, 0]
                obs[k, a, :, 1] = t*(s1[k, :] + self.load[k, a])
                s2 = self.dist_mat[:, :, int(self.shuttle_loc[k, a])]
                obs[k, a, :, 2] = t*s2[k, :]
                stat_coord[k, a, :, 0] = t*self.input_pnt[k, :, 0]
                stat_coord[k, a, :, 1] = t*self.input_pnt[k, :, 1]
                stat_ch_l[k, a, :, 0] = t*self.ch_l[k, :]
            
        obs = obs.astype(np.float32)
        stat_coord = stat_coord.astype(np.float32)
        stat_ch_l = stat_ch_l.astype(np.float32)
        return s, obs, self.avail_actions, baseline, reward, terminated, padded, EV_vec, charge_vec, current_time, stat_coord, stat_ch_l, sel_a, sh_vec 

