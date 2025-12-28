import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

# class LayerNormalization(nn.Module):
#     def __init__(self, eps=1e-6):
#         super(LayerNormalization, self).__init__()
#         self.eps = eps

#     def build(self, input_dim, device):
#         self.gamma = nn.Parameter(torch.ones(input_dim)).to(device)
#         self.beta = nn.Parameter(torch.zeros(input_dim)).to(device)

#     def forward(self, x):
#         if not hasattr(self, "gamma") or not hasattr(self, "beta"):
#             self.build(x.size(-1), x.device)
#         mean = x.mean(dim=-1, keepdim=True)
#         std = x.std(dim=-1, keepdim=True)
#         return self.gamma * (x - mean) / (std + self.eps) + self.beta

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = np.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.qs_linear = TimeDistributed(nn.Linear(d_model, n_head * d_k, bias=False), True)
        self.ks_linear = TimeDistributed(nn.Linear(d_model, n_head * d_k, bias=False), True)
        self.vs_linear = TimeDistributed(nn.Linear(d_model, n_head * d_v, bias=False), True)
        self.attention = ScaledDotProductAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.wo = TimeDistributed(nn.Linear(n_head * d_v, d_model), True)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        qs = self.qs_linear(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        ks = self.ks_linear(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        vs = self.vs_linear(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # Attention
        output, attn = self.attention(qs, ks, vs, mask)

        # Concatenate and apply final linear
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        output = self.dropout(self.wo(output))
        output = self.layer_norm(output + q)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Conv1d(d_model, d_inner, kernel_size=1)
        self.fc2 = nn.Conv1d(d_inner, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, x, mask=None):
        x, attn = self.self_attn(x, x, x, mask)
        x = self.pos_ffn(x)
        return x, attn

class Encoder(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(x)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns

def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return torch.tensor(pos_enc, dtype=torch.float)

def GetSubMask(s):
    len_s = s.size(1)
    mask = torch.tril(torch.ones((len_s, len_s), device=s.device))
    return mask.unsqueeze(0)

class DrrModel(nn.Module):
    def __init__(self, seq_len, d_feature, d_model=64, d_inner_hid=128, n_head=1, d_k=64, d_v=64, layers=2, dropout=0.1, model_type = 0, pos_mode = 0):
        super(DrrModel, self).__init__()
        self.seq_len = seq_len
        self.d_feature = d_feature
        self.d_model = d_model
        self.model_type = model_type
        if model_type == 1:
            self.uid_embedding = nn.Embedding(750000, 16) # for uid
            self.itemid_embedding = nn.Embedding(7500000, 32) # for icf1
            self.f1_embedding = nn.Embedding(8, 2) # for ucf1
            self.f2_embedding = nn.Embedding(4, 2) # for ucf2 & icf3
            self.f3_embedding = nn.Embedding(8, 2) # for ucf3 & icf4
            self.f4_embedding = nn.Embedding(4, 2) # for icf5
            self.f5_embedding = nn.Embedding(256, 4) # icf2

        self.dense = TimeDistributed(nn.Linear(d_feature, d_model), True)
        if pos_mode == 0:
            self.pos_embedding = nn.Embedding(seq_len, d_model, _weight=GetPosEncodingMatrix(seq_len, d_model), _freeze=True)
        elif pos_mode == 1:
            self.pos_embedding = nn.Embedding(seq_len, d_model)
        else:
            self.pos_embedding = None
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout)
        self.time_score_dense1 = TimeDistributed(nn.Linear(d_model, d_model))
        self.time_score_dense2 = TimeDistributed(nn.Linear(d_model, 1))

    def forward(self, v_input, pos_input, use_mask=False):
        if self.model_type == 0 or self.model_type == 2:
            d0 = self.dense(v_input)
        else:
            uid_input = v_input[:, :, 0].int()
            ucf1_input = v_input[:, :, 1].int()
            ucf2_input = v_input[:, :, 2].int()
            ucf3_input = v_input[:, :, 3].int()
            icf1_input = v_input[:, :, 4].int()
            icf2_input = v_input[:, :, 5].int()
            icf3_input = v_input[:, :, 6].int()
            icf4_input = v_input[:, :, 7].int()
            icf5_input = v_input[:, :, 8].int()
            v_input = v_input[:, :, 9:]
            #define user embedding
            u0 = self.uid_embedding(uid_input)
            u1 = self.f1_embedding(ucf1_input)
            u2 = self.f2_embedding(ucf2_input)
            u3 = self.f3_embedding(ucf3_input)
            #define item embedding
            i1 = self.itemid_embedding(icf1_input)
            i2 = self.f5_embedding(icf2_input)
            i3 = self.f2_embedding(icf3_input)
            i4 = self.f3_embedding(icf4_input)
            i5 = self.f4_embedding(icf5_input)
            #define page embedding: 16+2+2+2+32+4+2+2+2=64
            page_embedding = torch.cat([v_input, u0, u1, u2, u3, i1, i2, i3, i4, i5], dim=-1)
            d0 = self.dense(page_embedding)
        if self.pos_embedding is not None:
            p0 = self.pos_embedding(pos_input).squeeze(2)
            combine_input = d0 + p0
        else:
            combine_input = d0
        if use_mask:
            sub_mask = GetSubMask(pos_input)
        else:
            sub_mask = None

        enc_output, attn_weights = self.encoder(combine_input, mask=sub_mask)
        time_score = self.time_score_dense1(enc_output)
        time_score = torch.tanh(time_score)
        time_score = self.time_score_dense2(time_score).squeeze(-1)
        score_output = F.softmax(time_score, dim=-1)
        return score_output, attn_weights

class LRSchedulerPerStep:
    """
    Adjusts the learning rate per step based on the formula:
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup^(-1.5))

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for which to adjust the learning rate.
        d_model (int): Dimensionality of the model (used in learning rate scaling).
        warmup (int): Number of warmup steps.
    """
    def __init__(self, optimizer, d_model, warmup=4000):
        self.optimizer = optimizer
        self.basic = d_model**-0.5
        self.warm = warmup**-1.5
        self.step_num = 0

    def step(self):
        """Updates the learning rate based on the current step."""
        self.step_num += 1
        lr = self.basic * min(self.step_num**-0.5, self.step_num * self.warm)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class LRSchedulerPerEpoch:
    """
    Adjusts the learning rate per epoch based on the formula:
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup^(-1.5))

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for which to adjust the learning rate.
        d_model (int): Dimensionality of the model (used in learning rate scaling).
        warmup (int): Number of warmup steps.
        num_per_epoch (int): Number of steps per epoch.
    """
    def __init__(self, optimizer, d_model, warmup=4000, num_per_epoch=1000):
        self.optimizer = optimizer
        self.basic = d_model**-0.5
        self.warm = warmup**-1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def epoch_step(self):
        """Updates the learning rate at the start of each epoch."""
        self.step_num += self.num_per_epoch
        lr = self.basic * min(self.step_num**-0.5, self.step_num * self.warm)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr