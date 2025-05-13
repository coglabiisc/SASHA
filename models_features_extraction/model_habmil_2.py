from email.policy import strict

import torch
import torch.nn as nn
import torch.nn.functional as F

from clam.models.model_clam import Attn_Net_Gated, Attn_Net, initialize_weights
import numpy as np

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class HABMIL2(nn.Module):

    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embedding_size=1024, dropout_rate=0,
                 hidden_size=256,
                 temp=1):

        super(HABMIL2, self).__init__()
        self.size_dict = {"small": [embedding_size, 512, hidden_size], "big": [embedding_size, 512, 384]}
        size = self.size_dict[size_arg]

        # fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc = [nn.Linear(size[0], size[0]), nn.ReLU()]

        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            self.attention_net_1 = Attn_Net_Gated(L=size[0], D=size[1], dropout=dropout, n_classes=1)
            self.attention_net_2 = Attn_Net_Gated(L=size[0], D=size[1], dropout=dropout, n_classes=1)
        else:
            self.attention_net_1 = Attn_Net(L=size[0], D=size[1], dropout=dropout, n_classes=1)
            self.attention_net_2 = Attn_Net(L=size[0], D=size[1], dropout=dropout, n_classes=1)

        self.linear = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[0], n_classes)
        instance_classifiers = [nn.Linear(size[0], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.temp = temp
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net_1 = self.attention_net_1.to(device)
        self.attention_net_2 = self.attention_net_2.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device, dtype=torch.long)

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device, dtype=torch.long)

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):

        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        # all_instances = all_instances.to(device)
        # classifier.to(device)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        self.instance_loss_fn.to(device)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def load_pretrained_weights_from_clam(self, clam_model, clam_model_path, device):
        clam_model.load_state_dict(torch.load(clam_model_path, weights_only= True, map_location= device), strict = False)
        print(clam_model)
        self.linear[0].weight =  clam_model.attention_net[0].weight
        self.attention_net_1 = clam_model.attention_net[3]
        self.attention_net_2 = clam_model.attention_net[3]
        self.classifiers = clam_model.classifiers
        print("Model is initialized with pretrained weights from CLAM")
        return

    def freeze_initial_weights(self):

        self.linear.requires_grad_(False)
        self.attention_net_1.requires_grad_(False)
        self.attention_net_2.requires_grad_(False)
        return

    def forward(self, h, label=None, instance_eval=False, return_intermediate_features=False, attention_only=False):

        # Ip Format : NxKxd

        # Storing the initial shape
        num_patches = h.shape[0]
        map_factor = h.shape[1]

        # Step 1 ---> Attention N/w 1

        h = self.linear(h) # Op : N x d//2 ---> Will be denoted as d in further comments
        A, h = self.attention_net_1(h)  # Op : 1 ---> Attention Weights Nxkx1 , 2 ---> Input Again (Nxkxd)
        A = torch.transpose(A, -2, -1)  # Op : Nx1xk
        A = F.softmax(A, dim=-1) # Softmax over K # Op : Nxopxk

        attent_1_output = torch.bmm(A, h) # Op : N x op x d ---> Weighted summation
        h = attent_1_output.squeeze() # Op : N x d -----> Input for next attention

        if return_intermediate_features :
            return h

        # Step 2 ---> Attention N/w 2
        A, h = self.attention_net_2(h) # Op : 1 ---> Attention Weights Nx1 , 2 ---> Input Again (Nxd)
        A = torch.transpose(A, -1, -2) # Op : 1 x N
        A = F.softmax(A, dim=1)  # softmax over N

        if attention_only:
            return A
        A_raw = A

        if instance_eval:

            total_inst_loss = 0.0

            all_preds = []
            all_targets = []

            inst_labels = F.one_hot(label,
                                    num_classes=self.n_classes).squeeze()  # Converting label to one hot representation
            # Converting 0, 1 ---> [1, 0] / [0, 1]

            for i in range(len(self.instance_classifiers)):

                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:  # in-the-class : [Invoked when i == true label in one hot representation]

                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                else:  # out-of-the-class

                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:  # We have not subtyping ----> Continue from here
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:  # This part is false for out problem
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {'instance_loss': torch.tensor([0]).cuda(), 'inst_labels': np.array([0]),
                            'inst_preds': np.array([0])}

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def get_prediction_after_attention_network1(self, x, label= None, instance_eval=True, attention_only= False):

        A, h = self.attention_net_2(x)  # Op : 1 ---> Attention Weights Nx1 , 2 ---> Input Again (Nxd)

        if attention_only:
            return A

        A = torch.transpose(A, -1, -2)  # Op : 1 x N
        A = F.softmax(A, dim=1)  # softmax over N

        A_raw = A

        if instance_eval:

            total_inst_loss = 0.0

            all_preds = []
            all_targets = []

            inst_labels = F.one_hot(label,
                                    num_classes=self.n_classes).squeeze()  # Converting label to one hot representation
            # Converting 0, 1 ---> [1, 0] / [0, 1]

            for i in range(len(self.instance_classifiers)):

                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:  # in-the-class : [Invoked when i == true label in one hot representation]

                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                else:  # out-of-the-class

                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:  # We have not subtyping ----> Continue from here
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:  # This part is false for out problem
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, x)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {'instance_loss': torch.tensor([0]).cuda(), 'inst_labels': np.array([0]),
                            'inst_preds': np.array([0])}

        return logits, Y_prob, Y_hat, A_raw, results_dict

if __name__ == '__main__' :
    model = HABMIL2()
    # input = torch.randn((500, 4, 1024))
    # model(input, instance_eval=True, label = torch.tensor(1))
    model.load_pretrained_weights_from_clam()