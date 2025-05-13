import math
import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from architecture.network import Classifier_1fc, DimReduction, DimReduction1
from einops import repeat
from .nystrom_attention import NystromAttention
from modules.emb_position import *

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP_single_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_single_layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class ACMIL_MHA(nn.Module):
    def __init__(self, conf, n_token=1, n_masked_patch=0, mask_drop=0):
        super(ACMIL_MHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.sub_attention = nn.ModuleList()
        for i in range(n_token):
            self.sub_attention.append(MutiHeadAttention(conf.D_inner, 8, n_masked_patch=n_masked_patch, mask_drop=mask_drop))
        self.bag_attention = MutiHeadAttention_modify(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class

        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)

        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))

        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns


class MHA(nn.Module):
    def __init__(self, conf):
        super(MHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MutiHeadAttention(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        feat, attn = self.attention(q, k, v)
        output = self.classifier(feat)

        return output


class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and self.training:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class MutiHeadAttention_modify(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, v: Tensor, attn: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(v)

        # Separate into heads
        v = self._separate_heads(v, self.num_heads)

        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0]


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


class ABMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(ABMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = self.classifier(afeat)
        return outputs




class ACMIL_GA(nn.Module):
    def __init__(self, conf, D=128, droprate=0, n_token=1, n_masked_patch=0, mask_drop=0):
        super(ACMIL_GA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, n_token)
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = mask_drop


    def forward(self, x): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and self.training:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0)

    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat



class HACMIL_GA(nn.Module):
    def __init__(self, conf, D=128, droprate=0, n_token_1=1, n_token_2=1, n_masked_patch_1=0 ,n_masked_patch_2=0, mask_drop=0):
        super(HACMIL_GA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.dimreduction_2 = DimReduction(conf.D_feat, conf.D_inner)
        self.classifier = nn.ModuleList()
        for i in range(n_token_2):
            if conf.dim_reduction:
                self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
            else:
                self.classifier.append(Classifier_1fc(conf.D_feat, conf.n_class, droprate))
        self.n_masked_patch_1 = n_masked_patch_1
        self.n_masked_patch_2 = n_masked_patch_2
        self.n_token_1 = conf.n_token_1
        if conf.dim_reduction:
            self.attention_1 = Attention_Gated(conf.D_inner, D, n_token_1)
            self.attention_2 = Attention_Gated(conf.D_inner, D, n_token_2)
            self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate) 
        else:
            self.attention_1 = Attention_Gated(conf.D_feat, D, n_token_1)
            self.attention_2 = Attention_Gated(conf.D_feat, D, n_token_2)
            self.Slide_classifier = Classifier_1fc(conf.D_feat, conf.n_class, droprate)
        self.mask_drop = mask_drop
        self.use_dim_reduction = conf.dim_reduction


    def forward(self, x, extract_feature=False): ## x: N x 16 x 1024
        feat = x[0]
        x = self.dimreduction(feat) if self.use_dim_reduction else feat
        A_1 = self.attention_1(x).transpose(0,2).transpose(0,1)  ## n_token x N x 16


        if self.n_masked_patch_1 > 0 and self.training:
            # Get the indices of the top-k largest values
            N, n_token_1, k = A_1.shape #N x num_models x 16 , weigths across 16
            n_masked_patch = min(self.n_masked_patch_1, k)
            _, indices = torch.topk(A_1, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[
                torch.arange(indices.shape[0]).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.shape[1], rand_selected.shape[2]),  # Shape: [747, 2, 2]
                torch.arange(indices.shape[1]).unsqueeze(0).unsqueeze(-1).expand(indices.shape[0], -1, rand_selected.shape[2]),  # Shape: [747, 2, 2]
                rand_selected  # Shape: [747, 2, 2]
            ]
            random_mask = torch.ones(N, n_token_1, k).to(A_1.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A_1 = A_1.masked_fill(random_mask == 0, -1e9)

        A_1 = F.softmax(A_1, dim=-1)  # softmax over 16
        bag_A1 = A_1.mean(dim=1, keepdim=True)
        afeat_1 = torch.bmm(bag_A1, feat).squeeze(1) ## K x L
        y = self.dimreduction_2(afeat_1) if self.use_dim_reduction else afeat_1
        A_2 = self.attention_2(y)
        if self.n_masked_patch_2 > 0 and self.training:
            k,n = A_2.shape
            n_masked_patch = min(self.n_masked_patch_2, n)
            _, indices = torch.topk(A_2, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A_2.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A_2 = A_2.masked_fill(random_mask == 0, -1e9)
        attn_raw = A_2
        A_2 = F.softmax(A_2, dim=1)
        afeat_2 = torch.mm(A_2, afeat_1)
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat_2[i]))
        bag_A = A_2.mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, afeat_1)
        return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_1, A_2, afeat_1, attn_raw


    # def forward(self, x, extract_feature=False): ## x: N x 16 x 1024
    #     feat = x[0]
    #     x = self.dimreduction(feat) if self.use_dim_reduction else feat
    #     A_1 = self.attention_1(x).transpose(0,2).transpose(0,1)  ## n_token x N x 16


    #     if self.n_masked_patch_1 > 0 and self.training:
    #         # Get the indices of the top-k largest values
    #         N, n_token_1, k = A_1.shape #N x num_models x 16 , weigths across 16
    #         n_masked_patch = min(self.n_masked_patch_1, k)
    #         _, indices = torch.topk(A_1, n_masked_patch, dim=-1)
    #         rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :, :int(n_masked_patch * self.mask_drop)]
    #         masked_indices = indices[
    #             torch.arange(indices.shape[0]).unsqueeze(-1).unsqueeze(-1).expand(-1, indices.shape[1], rand_selected.shape[2]),  # Shape: [747, 2, 2]
    #             torch.arange(indices.shape[1]).unsqueeze(0).unsqueeze(-1).expand(indices.shape[0], -1, rand_selected.shape[2]),  # Shape: [747, 2, 2]
    #             rand_selected  # Shape: [747, 2, 2]
    #         ]
    #         random_mask = torch.ones(N, n_token_1, k).to(A_1.device)
    #         random_mask.scatter_(-1, masked_indices, 0)
    #         A_1 = A_1.masked_fill(random_mask == 0, -1e9)

    #     A_1 = F.softmax(A_1, dim=-1)  # softmax over 16
    #     bag_A1 = A_1.mean(dim=1, keepdim=True)
    #     afeat_1 = torch.bmm(bag_A1, feat).squeeze(1) ## K x L
    #     y = self.dimreduction_2(afeat_1) if self.use_dim_reduction else afeat_1
    #     A_2 = self.attention_2(y)
    #     A_2NS = A_2[:5, :]
    #     A_2S_raw = A_2[5:, :]
    #     if self.n_masked_patch_2 > 0 and self.training:
    #         k,n = A_2NS.shape
    #         n_masked_patch = min(self.n_masked_patch_2, n)
    #         _, indices = torch.topk(A_2NS, n_masked_patch, dim=-1)
    #         rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
    #         masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
    #         random_mask = torch.ones(k, n).to(A_2NS.device)
    #         random_mask.scatter_(-1, masked_indices, 0)
    #         A_2NS = A_2NS.masked_fill(random_mask == 0, -1e9)
    #     A_2NS = self.alpha * A_2NS
    #     A_2S = (1 - self.alpha) * A_2S_raw
    #     A_2_raw = torch.cat((A_2NS, A_2S), dim=0)
    #     A_2 = F.softmax(A_2, dim=1)
    #     afeat_2 = torch.mm(A_2, afeat_1)
    #     outputs = []
    #     for i, head in enumerate(self.classifier):
    #         outputs.append(head(afeat_2[i]))
    #     bag_A = A_2.mean(0, keepdim=True)
    #     bag_feat = torch.mm(bag_A, afeat_1)
    #     return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_1, A_2, afeat_1, A_2_raw
    


    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat
    

    def classify(self, x, average_block2_weights=False):
        feat = x[0]
        y = self.dimreduction_2(feat) if self.use_dim_reduction else feat
        A_2 = self.attention_2(y)
        if self.n_masked_patch_2 > 0 and self.training:
            k,n = A_2.shape
            n_masked_patch = min(self.n_masked_patch_2, n)
            _, indices = torch.topk(A_2, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A_2.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A_2 = A_2.masked_fill(random_mask == 0, -1e9)
        A_2 = F.softmax(A_2, dim=1)
        # afeat_2 = torch.mm(A_2, feat)
        # outputs = []
        # for i, head in enumerate(self.classifier):
        #     outputs.append(head(afeat_2[i]))
        bag_A = A_2.mean(0, keepdim=True)

        if average_block2_weights:
            return bag_A

        bag_feat = torch.mm(bag_A, feat)
        return self.Slide_classifier(bag_feat), A_2