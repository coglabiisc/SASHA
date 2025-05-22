import torch
import torch.nn as nn


class WSIObservationEnv():
    """
    ### Description
    Environment for RLogist with discrete action space and feature update mechanism.

    ### Observation Space
    The observation is a `ndarray` with shape `(REGION_NUM, EMBEDDING_LENGTH)` where the elements correspond to the following:
    | Num | Observation            | Min  | Max |
    |-----|------------------------|------|-----|
    | 0   | Index of the region    | 0    | REGION_NUM-1 |
    | 1   | Scanning-level feature | -Inf | Inf |

    ### Action Space
    There is 1 discrete deterministic actions:
    | Num | Observation                                     | Value    |
    |-----|-------------------------------------------------|----------|
    | 0   | the index of target region for analyze in depth | [0, REGION_NUM-1] |

    ### Transition Dynamics:
    Given an action, the WSIObservationEnv follows the following transition dynamics:
    Update the scanning-level feature of all unobserved regions with f_local and f_global.

    ### Reward:
    The goal is to predict the slide-level label, as such the agent assigned with a reward of 1 for the
    right classification result.

    ### Starting State
    The scanning-level feature is extracted with pretrained models i.e. ResNet50

    ### Episode End
    The episode ends if the length of the episode reaches MAX_LENGTH.
    """

    def __init__(self, lr_features, hr_features, label, conf):
        self.conf = conf
        self.features = lr_features
        self.current_state = lr_features
        self.hr_features = hr_features
        self.label = label
        self.B, self.N, self.d = self.current_state.shape
        self.current_time_step = 0
        self.max_time_steps = int(self.N * conf.frac_visit)
        self.visited_patches = torch.zeros(size=(self.N,), dtype=torch.bool)
        self.visited_patch_idx = []
        self.reward = []


    def reset(self):
        self.current_state = self.features
        self.current_time_step = 0
        self.visited_patches = torch.zeros(size=(self.N,), dtype=torch.bool)
        self.visited_patch_idx = []
        self.reward = []
        return self.current_state
    
    @torch.no_grad()
    def step(self, action, state_update_net, classifier_net, device):
        
        if self.current_time_step < self.max_time_steps:
            self.current_time_step += 1
            self.visited_patches[action] = 1
            self.visited_patch_idx.append(action.item())

            # Making changes for the fglobal part
            z_at = self.current_state[0][action].repeat(self.N, 1)  # 1xd -> Nxd
            v_at = self.hr_features[action].repeat(self.N, 1)  # 1xd -> Nxd
            ip = torch.concat((z_at, v_at, self.current_state[0]), dim=-1)  # Nx3d
            ip = ip.unsqueeze(dim= 0)

            # Updating to new state
            new_state = state_update_net(ip)

            # Now updating the visited patch index with actual H.R. representation
            new_state[0][self.visited_patch_idx] = self.hr_features[self.visited_patch_idx]
            slide_preds, attn = classifier_net.classify(new_state)
            pred = torch.softmax(slide_preds, dim=-1)
            loss = nn.CrossEntropyLoss()(pred, self.label)
            self.reward.append(-loss.item())
            if self.current_time_step > 1:
                reward = self.reward[-1] - self.reward[-2]
            else:
                reward = -loss.item()
            self.current_state = new_state
            done = False


            # update the state
            # check whether all the patches are visited
            # if so return the final state with reward
            if self.visited_patches.sum().item() == self.N or self.current_time_step == self.max_time_steps:
                slide_preds, attn = classifier_net.classify(new_state)
                pred = torch.softmax(slide_preds, dim=-1)
                loss = nn.CrossEntropyLoss()(pred, self.label)
                self.reward.append(-loss.item())
                reward = self.reward[-1]
                done = True
                self.reset()

        self.current_state = self.features
        new_state = self.current_state
        return new_state, reward, done
        