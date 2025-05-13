from types import SimpleNamespace

import torch.nn as nn
import torch
import torch.nn.functional as F


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
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
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        return A  ### K x N


class HACMIL_GA_Sparse(nn.Module):
    def __init__(self, conf, D=128, droprate=0, n_token_1=1, n_token_2=1, n_token_3 = 1, n_masked_patch_1=0, n_masked_patch_2=0,
                 mask_drop=0):
        super(HACMIL_GA_Sparse, self).__init__()
        self.n_token_1 = n_token_1
        self.n_token_2 = n_token_2
        self.n_token_3 = n_token_3

        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.dimreduction_2 = DimReduction(conf.D_feat, conf.D_inner)

        self.classifier = nn.ModuleList()
        for i in range(n_token_2 + n_token_3):
            if conf.dim_reduction:
                self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
            else:
                self.classifier.append(Classifier_1fc(conf.D_feat, conf.n_class, droprate))

        self.n_masked_patch_1 = n_masked_patch_1
        self.n_masked_patch_2 = n_masked_patch_2

        if conf.dim_reduction:
            self.attention_1 = Attention_Gated(conf.D_inner, D, n_token_1)
            self.attention_2 = Attention_Gated(conf.D_inner, D, n_token_2 + n_token_3)
            self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        else:
            self.attention_1 = Attention_Gated(conf.D_feat, D, n_token_1)
            self.attention_2 = Attention_Gated(conf.D_feat, D, n_token_2 + n_token_3)
            self.Slide_classifier = Classifier_1fc(conf.D_feat, conf.n_class, droprate)
        self.mask_drop = mask_drop
        self.use_dim_reduction = conf.dim_reduction
        self.attention_3 = Attention_Gated(conf.D_feat, D, 1)

    def forward(self, x, extract_feature= False):  ## x: N x 16 x 1024

        feat = x[0]
        x = self.dimreduction(feat) if self.use_dim_reduction else feat
        A_1 = self.attention_1(x).transpose(0, 2).transpose(0, 1)  ## n_token x N x 16

        if self.n_masked_patch_1 > 0 and self.training:
            # Get the indices of the top-k largest values
            N, n_token_1, k = A_1.shape  # N x num_models x 16 , weigths across 16
            n_masked_patch = min(self.n_masked_patch_1, k)
            _, indices = torch.topk(A_1, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :,
                            :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[
                torch.arange(indices.shape[0]).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.shape[1],
                                                                                  rand_selected.shape[
                                                                                      2]),  # Shape: [747, 2, 2]
                torch.arange(indices.shape[1]).unsqueeze(0).unsqueeze(-1).expand(indices.shape[0], -1,
                                                                                 rand_selected.shape[
                                                                                     2]),  # Shape: [747, 2, 2]
                rand_selected  # Shape: [747, 2, 2]
            ]
            random_mask = torch.ones(N, n_token_1, k).to(A_1.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A_1 = A_1.masked_fill(random_mask == 0, -1e9)

        A_1 = F.softmax(A_1, dim=-1)  # softmax over 16
        bag_A1 = A_1.mean(dim=1, keepdim=True)
        afeat_1 = torch.bmm(bag_A1, feat).squeeze(1)  ## K x L

        if extract_feature :
            return afeat_1

        y = self.dimreduction_2(afeat_1) if self.use_dim_reduction else afeat_1
        A_2 = self.attention_2(y)
        if self.n_masked_patch_2 > 0 and self.training:
            k, n = A_2.shape
            n_masked_patch = min(self.n_masked_patch_2, n)
            _, indices = torch.topk(A_2, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A_2.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A_2 = A_2.masked_fill(random_mask == 0, -1e9)

        A_2_raw = A_2
        A_2 = F.softmax(A_2, dim=1)
        afeat_2 = torch.mm(A_2, afeat_1)
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat_2[i]))

        # Original
        # bag_A = A_2.mean(0, keepdim=True)
        # bag_feat = torch.mm(bag_A, afeat_1)
        # return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_1, A_2, afeat_1

        # Changes for the prescribe model of sparse and non sparse  [[[ Option 1 ]]]

        bag_A_non_sparse= A_2[:self.n_token_2].mean(0, keepdim=True)
        bag_A_sparse = A_2[self.n_token_2:].mean(0, keepdim=True)

        bag_feat_non_sparse = torch.mm(bag_A_non_sparse, afeat_1)  # Op : 1xd
        bag_feat_sparse = torch.mm(bag_A_sparse, afeat_1)  # Op : 1xd

        bag_feat = torch.concat((bag_feat_sparse, bag_feat_non_sparse), dim=0)  # Op : 2xd
        A_3 = self.attention_3(bag_feat)
        A_3 = F.softmax(A_3, dim=1)
        feat = torch.mm(A_3, bag_feat)

        return torch.stack(outputs, dim=0), self.Slide_classifier(feat), A_1, A_2[:self.n_token_2], A_2[self.n_token_2:], A_2

    def classify(self, x):

        afeat_1 = x[0]
        y = self.dimreduction_2(afeat_1) if self.use_dim_reduction else afeat_1
        A_2 = self.attention_2(y)
        if self.n_masked_patch_2 > 0 and self.training:
            k, n = A_2.shape
            n_masked_patch = min(self.n_masked_patch_2, n)
            _, indices = torch.topk(A_2, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A_2.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A_2 = A_2.masked_fill(random_mask == 0, -1e9)

        A_2_raw = A_2
        A_2 = F.softmax(A_2, dim=1)
        afeat_2 = torch.mm(A_2, afeat_1)
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat_2[i]))

        # Changes for the prescribe model of sparse and non sparse  [[[ Option 1 ]]]
        bag_A_sparse = A_2[:self.n_token_2 // 2].mean(0, keepdim=True)
        bag_A_non_sparse = A_2[self.n_token_2 // 2:].mean(0, keepdim=True)

        bag_feat_sparse = torch.mm(bag_A_sparse, afeat_1)  # Op : 1xd
        bag_feat_non_sparse = torch.mm(bag_A_non_sparse, afeat_1)  # Op : 1xd
        bag_feat = torch.concat((bag_feat_sparse, bag_feat_non_sparse), dim=0)  # Op : 2xd
        A_3 = self.attention_3(bag_feat)
        A_3 = F.softmax(A_3, dim=1)
        feat = torch.mm(A_3, bag_feat)

        return self.Slide_classifier(feat), A_2


@torch.no_grad()
def load_model(ckpt_path, device):
    dict = torch.load(ckpt_path, map_location=device)
    curr_epoch = dict['epoch']
    config = dict['config']
    model_dict = dict['model']
    optimizer_dict = dict['optimizer']
    return model_dict, optimizer_dict, config, curr_epoch
