from collections import namedtuple

# please specify here the network structure for FFEVSS
# input_dim = 4 that include x, y coords, # EVs and charging levels 
# n_nodes - total number of nodes in the network 
# n_agents - # of shuttles deployed 
# n_drivers - # of drivers at each shuttle 
# decode-len - # of time steps used for training 
# capacity - # of seating available at each shuttle 
# n-charge - # of charger nodes 
# n_demander - # of demander nodes 
# n_supplier = n_nodes-1-(n_charger+n_demander)
# the last node is reserved for a depot. 

NetworkFFEVSS = namedtuple('NetworkFFEVSS', ['network_name', 
						'input_dim',
						'n_nodes' ,
                        'n_agents',
						'n_drivers',
						'decode_len',
						'capacity',
                        'n_charge', 
                        'n_demand', 
                        'difficulty'])



network_lst = {}



FFEVSS23 = NetworkFFEVSS(network_name = 'FFEVSS23',
			  input_dim=4,
			  n_nodes = 23,
              n_agents = 1,  
			  n_drivers = 3,
			  decode_len= 45,
			  capacity=6,
            n_charge= 7, 
            n_demand = 7, 
            difficulty='easy')
network_lst['FFEVSS23'] = FFEVSS23