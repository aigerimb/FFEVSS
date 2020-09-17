import argparse
from network_settings import network_lst 


def str2bool(v):
    return v.lower() in ('true', '1')

def add_network_settings(args,problem):

    try:
        network_params = network_lst[problem]
    except:
        raise Exception('Task is not implemented.') 

    for name, value in network_params._asdict().items():
    	args[name] = value
    return args

def ParseParams():
    parser = argparse.ArgumentParser(description="FFEVSS Rebalancing")

    # Data generation for Training and Testing 
    parser.add_argument('--network', default='FFEVSS23', help="Define an urabn network structure")
    parser.add_argument('--batch_size', default=128,type=int, help='Batch size for training')
    parser.add_argument('--n_train', default=20000,type=int, help='# of episodes for training')
    parser.add_argument('--test_size', default=128,type=int, help='# of instances for testing')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_interval', default=200,type=int, help='test every test_interval steps')
    parser.add_argument('--save_interval', default=1000,type=int, help='save every save_interval steps')
    
    # Neural Network Structure 
    
    # Embedding 
    parser.add_argument('--embedding_dim', default=3,type=int, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128,type=int, help='Dimension of hidden layers in Enc/Dec')
    
    # Decoder: LSTM 
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of LSTM layers in the encoder and decoder')
    parser.add_argument('--forget_bias', default=1.0,type=float, help="Forget bias for BasicLSTMCell.")
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout prob')
    parser.add_argument('--decode_len', default=None,type=int,                     
                        help='Number of time steps the decoder runs before stopping')
    # Attention 
    parser.add_argument('--use_tanh', type=str2bool, default=False, help='use tahn before computing probs in attention')
    parser.add_argument('--mask_logits', type=str2bool, default=True, help='mask unavailble nodes probs')
    
    # Training
    parser.add_argument('--train', default=True,type=str2bool, help="whether to do the training or not")
    parser.add_argument('--actor_net_lr', default=1e-4,type=float, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4,type=float, help="Set the learning rate for the critic network")
    parser.add_argument('--random_seed', default= 5,type=int, help='')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='Gradient clipping')
  
    args, unknown = parser.parse_known_args()
    args = vars(args)
    args = add_network_settings(args,args['network'])
    
    for key, value in sorted(args.items()):
        print("{}: {}".format(key,value))
    
    return args 