import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cpu')

class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)



class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, use_tahn=False, C = 10):
        super(Attention, self).__init__()
        self.use_tahn = use_tahn 
        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.project_d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_d_rem = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_dist = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_ch_l = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_ref = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.C = C

    def forward(self, static_hidden, static_ch_l, dynamic_hidden, decoder_hidden):
        # [b_s, hidden_dim, n_nodes]
        d, d_rem, dist = dynamic_hidden 
        batch_size, hidden_size, n_nodes = static_hidden.size()
      
        # [b_s, hidden_dim, n_nodes]
        d_ex = self.project_d(d)
        d_rem = self.project_d_rem(d_rem)
        dist = self.project_dist(dist)
        ch_l = self.project_ch_l(static_ch_l)
        # [b_s, hidden_dim, n_nodes]
        e = self.project_ref(static_hidden)
        # [b_s, hidden_dim]
        decoder_hidden = self.project_query(decoder_hidden)
       
        
        
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(batch_size, hidden_size, n_nodes)
       

        u = torch.bmm(v, torch.tanh(e + q + d_ex + d_rem + ch_l + dist)).squeeze(1)
        if self.use_tahn:
            logits = self.C * self.tanh(u)
        else:
            logits = u 
        # e : [b_s, hidden_dim, n_nodes]
        # logits : [b_s, n_nodes]
        return e, logits 
    
    

class AttentionCritic(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, use_tahn=False, C = 10):
        super(AttentionCritic, self).__init__()
        self.use_tahn = use_tahn 
        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.project_d_ex = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_ch_l = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_ref = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.C = C

    def forward(self, static_hidden, static_ch_l, dynamic_hidden, decoder_hidden):
        # [b_s, hidden_dim, n_nodes]
        
        batch_size, hidden_size, n_nodes = static_hidden.size()
      
        # [b_s, hidden_dim, n_nodes]
        d_ex = self.project_d_ex(dynamic_hidden)
        ch_l = self.project_ch_l(static_ch_l)
        # [b_s, hidden_dim, n_nodes]
        e = self.project_ref(static_hidden)
        # [b_s, hidden_dim]
        decoder_hidden = self.project_query(decoder_hidden)
       
        
        
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(batch_size, hidden_size, n_nodes)
       

        u = torch.bmm(v, torch.tanh(e + q + d_ex + ch_l)).squeeze(1)
        if self.use_tahn:
            logits = self.C * self.tanh(u)
        else:
            logits = u 
        # e : [b_s, hidden_dim, n_nodes]
        # logits : [b_s, n_nodes]
        
        return e + ch_l, logits 


class Decoder(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1, n_glim=0):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bias = False, 
                        batch_first=True,bidirectional=False, dropout=dropout if num_layers > 1 else 0)

        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)
        

    def forward(self, static_hidden, static_ch_l, dynamic_hidden, decoder_input, last_hh):
        # decoder_input: [b_s, hidden_dim, 1]
        # rnn_out : [b_s, hidden_dim]
        # last_hh : [num_layers, b_s, hidden_dim]
        
        rnn_out, last_hh = self.lstm(decoder_input.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            hx = self.drop_hh(last_hh[0]) 
            cx = self.drop_hh(last_hh[1]) 
            last_hh = (hx, cx)
        #[b_s, hidden_dim]
        hy = last_hh[0].squeeze(0)
       

        # compute attention 
        _, logits = self.encoder_attn(static_hidden, static_ch_l, dynamic_hidden, hy)
        
     
        return logits, last_hh


class Actor(nn.Module):
   

    def __init__(self, hidden_size,  
                 num_layers=1, dropout=0.1, mask_logits=True):
        super(Actor, self).__init__()
        
        self.mask_logits = mask_logits 
        # Define the encoder & decoder models
        # for static x, y coords 
        self.static_encoder = Encoder(2, hidden_size)
        self.static_ch_l = Encoder(1, hidden_size)
        self.dynamic_d_ex = Encoder(1, hidden_size)
        self.dynamic_d_rem = Encoder(1, hidden_size)
        self.dynamic_dist = Encoder(1, hidden_size)
        
        self.decoder = Decoder(hidden_size, num_layers, dropout)
        self.logsoft = nn.LogSoftmax()
        self.Bignumber = 100000

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
                
    def emd_stat(self, static, ch_l):
        
        return self.static_encoder(static), self.static_ch_l(ch_l)


    def forward(self, static_hidden, static_ch_l, dynamic, decoder_input, last_hh, terminated, avail_actions):
        
        # static_hidden: embedding of x,y coords [b_s, hidden_dim, n_nodes]
        # static_ch_l : embedding of charging levels 
        # decoder_input: embedding of x, y coords of the last selected node [b_s, hidden_dim, 1]
        # last_hh: hidden state of lstm [num_layers, batch_size, hidden_dim]
        # dynamic: current state [b_s, n_nodes, dynamic_size]
        
        #[b_s, 1, n_nodes] each 
        d_ex, d_rem, dist = dynamic.permute(0, 2, 1).chunk(3, dim=1)
      
        emb_d_ex = self.dynamic_d_ex(d_ex)
        emb_d_rem = self.dynamic_d_rem(d_rem)
        emb_dist = self.dynamic_dist(dist)
        dynamic_hidden = (emb_d_ex, emb_d_rem, emb_dist)
       
        logits, last_hh = self.decoder(static_hidden, static_ch_l, 
                                          dynamic_hidden,
                                          decoder_input, last_hh)
        if self.mask_logits:
            logits[avail_actions==0] = -self.Bignumber
        
        logprobs = self.logsoft(logits)
       # probs = F.softmax(probs + avail_actions.log(), dim=1)
        probs = torch.exp(logprobs)
  
            
        if self.training:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            logp = m.log_prob(action)
        else:
            prob, action = torch.max(probs, 1)  # Greedy
            logp = prob.log()

              
        logp = logp * (1. - terminated)

        return action, probs, logp, last_hh 

    
    
class Critic(nn.Module):
    """Estimates the problem complexity.
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size, n_processes=3, num_layers=1):
        super(Critic, self).__init__()
        
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.dynamic_d_ex = Encoder(1, hidden_size)
        self.static_encoder = Encoder(2, hidden_size)
        self.static_ch_l = Encoder(1, hidden_size)
        self.attention1 = AttentionCritic(hidden_size)
        self.attention2 = AttentionCritic(hidden_size)
        self.attention3 = AttentionCritic(hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, static_ch_l, d_ex):
        
        static_hidden = self.static_encoder(static)
        
        batch_size, _, n_nodes = static_hidden.size()
       
        dynamic_hidden = self.dynamic_d_ex(d_ex)
        static_ch_l =self.static_ch_l(static_ch_l)
       
        
        hx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hy = hx.squeeze(0)
        
        e, logits = self.attention1(static_hidden, static_ch_l, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        
        # [b_s, hidden_dim] = [b_s, 1, n_nodes] * [b_s, n_nodes, hidden_dims]
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
        e, logits = self.attention2(static_hidden, static_ch_l, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
        e, logits = self.attention3(static_hidden, static_ch_l, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
        
        out = F.relu(self.fc1(hy))
        out = self.fc2(out)
        
        return out 
    