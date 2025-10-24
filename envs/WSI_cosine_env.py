import math

import torch
import torch.nn as nn

### For Intermediate Reward

class WSICosineObservationEnv():
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
        self.high_cosine_indices = None
        self.conf = conf
        self.features = lr_features.clone()
        self.current_state = self.features.clone()
        self.hr_features = hr_features
        self.label = label
        self.B, self.N, self.d = self.current_state.shape
        self.current_time_step = 0
        self.max_time_steps = math.ceil(self.N * conf.frac_visit)
        self.visited_patches = torch.zeros(size=(self.N,), dtype=torch.bool).to(conf.device)
        self.visited_patch_idx = []
        self.reward = []
        self.mask = torch.ones((self.N,))
        self.cosine_threshold = conf.cosine_threshold
        self.only_ce_as_reward = conf.only_ce_as_reward
        self.visited_all_patches = False

    def reset(self):
        self.current_state = self.features.clone()
        self.current_time_step = 0
        self.visited_patches = torch.zeros(size=(self.N,), dtype=torch.bool)
        self.visited_patch_idx = []
        self.reward = []
        self.mask = torch.ones((self.N,))
        self.visited_all_patches = False
        # return self.current_state.clone()

    @torch.no_grad()
    def step(self, action, state_update_net, classifier_net, device):

        # Original Code
        if self.visited_all_patches:
            self.current_time_step += 1
            done = False
            slide_preds, attn = classifier_net.classify(self.current_state)
            loss = nn.CrossEntropyLoss()(slide_preds, self.label)
            self.reward.append(-loss.item())
            reward = -loss.item()
            if self.current_time_step == self.max_time_steps:
                done = True
                final_state = self.current_state.clone()
                self.reset()
                return final_state, reward, done
            else:
                return self.current_state, reward, done

        if self.current_time_step < self.max_time_steps:
            self.current_time_step += 1
            self.visited_patches[action] = 1
            self.visited_patch_idx.append(action.item())
            # getting patch indices which are not visited
            valid_indices = (self.mask == 1).nonzero()[:, 0]
            # Step 1 : Compute the cosine similarity with all states in low resolution
            cosine_vector = torch.cosine_similarity(self.current_state[0], self.current_state[0][action])
            # Step 2 : Now pick only those vector which is having cosine similarity  geq args.cosine_threshold
            high_cosine_indices = (torch.abs(cosine_vector) >= self.cosine_threshold).nonzero()[:, 0]
            # Step 3 : Masking the patches
            self.mask[high_cosine_indices] = 0
            self.visited_patches[high_cosine_indices] = 1
            # Step 4 : Now removing the indices of random_idx from it bcoz we have visited that patch
            high_cosine_indices = high_cosine_indices[high_cosine_indices != action]
            # Step 5 : Now only keep those values where the high cosine indices and patch idx is unmasked i.e. = 1
            high_cosine_indices = torch.tensor(list(set(high_cosine_indices.tolist()).intersection(valid_indices.tolist())), device= device)
            # High cosine indices
            self.high_cosine_indices = high_cosine_indices  # TODO : Added this line to store the logic ....
            v_at = self.hr_features[action].to(device)
            # Just updating the patch if no other patch have same cosine similarity
            if high_cosine_indices.shape[0] == 0:
                self.current_state[0][action] = v_at
            else:
                # Update the unobserved region of current state with f_global logic
                input_f_global = torch.cat((v_at.repeat(len(high_cosine_indices), 1),
                                            self.current_state[0][action].repeat(len(high_cosine_indices), 1),
                                            self.current_state[0][high_cosine_indices]), dim=1)
                self.current_state = self.current_state.clone()
                self.current_state[:, high_cosine_indices, :] = state_update_net(input_f_global)

            # new_state = self.current_state
            self.current_state[0][self.visited_patch_idx] = self.hr_features[self.visited_patch_idx]
            slide_preds, attn = classifier_net.classify(self.current_state)
            # pred = torch.softmax(slide_preds, dim=-1)
            loss = nn.CrossEntropyLoss()(slide_preds, self.label)
            self.reward.append(-loss.item())
            if self.current_time_step > 1 and not self.only_ce_as_reward:
                reward = self.reward[-1] - self.reward[-2]
            else:
                reward = - loss.item()
            # self.current_state = new_state
            done = False

            # check whether all the patches are visited
            # if so return the final state with reward
            if self.current_time_step == self.max_time_steps:
                done = True
                final_state = self.current_state.clone()
                self.reset()
                return final_state, reward, done

            if self.visited_patches.sum().item() == self.N:
                self.visited_all_patches = True

        return self.current_state.clone(), reward, done

    def get_similar_patches(self):
        return self.high_cosine_indices