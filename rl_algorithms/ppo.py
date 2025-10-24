import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from architecture.network import DimReduction


class Value_fc(nn.Module):
    def __init__(self, n_channels, droprate=0.0):
        super(Value_fc, self).__init__()
        self.fc = nn.Linear(n_channels, 1)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N
    

class Critic(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(Critic, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        if conf.dim_reduction:
            self.attention = Attention_Gated(conf.D_inner, D, 1)
            self.final_layer = Value_fc(conf.D_inner, droprate) 
        else:
            self.attention = Attention_Gated(conf.D_feat, D, 1)
            self.final_layer = Value_fc(conf.D_feat, droprate)
        self.use_dim_reduction = conf.dim_reduction


    def forward(self, x):
        state = x
        x = self.dimreduction(x) if self.use_dim_reduction else x
        A = self.attention(x).transpose(0,2).transpose(0,1)

        A = F.softmax(A, dim=-1) 
        h = torch.bmm(A, state)
        value = self.final_layer(h)
        return value
    

class Actor(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(Actor, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        if conf.dim_reduction:
            self.attention = Attention_Gated(conf.D_inner, D, 1)
        else:
            self.attention = Attention_Gated(conf.D_feat, D, 1)
        self.use_dim_reduction = conf.dim_reduction

    
    def forward(self, x):
        state = x
        x = self.dimreduction(x) if self.use_dim_reduction else x
        A = self.attention(x).transpose(0,2).transpose(0,1)

        # A = F.softmax(A, dim=-1) 
        return A
    

'''
Agnet: 
1. Actor - Input (N * d) state representation; Outputs a probability distribution over N
2. Critic - Input (N * d) state representation; Outputs the expected return from being in that state
'''

class Agent(nn.Module):
    def __init__(self, actor, critic, conf):
        super(Agent, self).__init__()
        self.conf = conf
        self.critic = critic
        self.actor = actor


    def get_value(self, x):
        value = self.critic(x)
        return value

    def get_action(self, x, visited_patch_ids, is_eval=False, is_top_k = False, is_top_p = False):
        A_raw = self.actor(x).view(1, -1)
        A_raw[0][visited_patch_ids] = -torch.inf
        dist = Categorical(logits=A_raw)


        # This code is added for random policy (Uncomment if need to execute this)
        # Sample Randomly from the patches --->
        # N = A_raw.shape[1]
        # all_indices = list(range(N))
        # remaining_indices = list(set(all_indices) - set(visited_patch_ids))
        # random_patch = random.choice(remaining_indices)
        # return torch.tensor(random_patch), None , None

        # Get action based on different sampling strategy ----
        # By default we will follow pick the max from probability distribution for SASHA inference

        if is_eval :
            # Now check which flag is up
            if is_top_k and is_top_p :
                raise ValueError("Only one of is_top_k or is_top_p can be True")

            elif is_top_k and not is_top_p :

                k = 3  # or 3 / 5 ----> determine top_k to be unmasked
                A_raw = A_raw.squeeze()
                topk_vals, topk_idxs = torch.topk(A_raw, k)
                probs = torch.zeros_like(A_raw)
                probs[topk_idxs] = torch.nn.functional.softmax(topk_vals, dim=0)
                dist = Categorical(probs=probs)
                action = dist.sample()

            elif not is_top_k and is_top_p:

                p = 0.3
                A_raw = A_raw.squeeze()
                sorted_logits, sorted_indices = torch.sort(A_raw, descending=True)
                sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=0)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Mask out all logits beyond the cumulative probability threshold
                cutoff_idx = torch.where(cumulative_probs > p)[0]
                if len(cutoff_idx) > 0:
                    last_idx = cutoff_idx[0].item() + 1
                else:
                    last_idx = len(A_raw)

                selected_indices = sorted_indices[:last_idx]
                probs = torch.zeros_like(A_raw)
                probs[selected_indices] = torch.nn.functional.softmax(A_raw[selected_indices], dim=0)
                dist = Categorical(probs=probs)
                action = dist.sample()

            else :
                action = torch.argmax(dist.probs)  # This is the by default choice for the inference mode

        else :
            action = dist.sample() # This is performed during training

        # Continue ....
        log_prob = dist.log_prob(action)

        return action, log_prob.detach(), dist.entropy()
    

    def get_action_and_value(self, x, visited_patch_ids):
        action = self.get_action(x, visited_patch_ids)
        value = self.get_value(x)
        return action, value


    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		
        """

		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()
        A_raw = self.actor(batch_obs)
        dist = Categorical(logits=A_raw.squeeze(1))
        log_prob = dist.log_prob(batch_acts)
		
        return V, log_prob, dist.entropy()

