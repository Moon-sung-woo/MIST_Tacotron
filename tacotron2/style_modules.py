# source from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, E, ref_enc_filters, n_mels):

        super().__init__()
        K = len(ref_enc_filters)
        filters = [3] + ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)])

        # n_mels, 3, 2, 1, K) #여기서 채널에 대한 계산이 이루어 지니깐 여기도 같이 바꿔주면 됨.
        out_channels = self.calculate_channels(480, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=E,
                          batch_first=True)
        self.n_mels = n_mels

    def forward(self, inputs):
        N = inputs.size(0)
        # 들어오는 이미지를 어떻게 처리해줄 건지에 대해서 생각해보고 넣어주면 됨.
        out = inputs.view(N, 3, -1, inputs.size(2))  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        _, out = self.gru(out)  # out --- [1, N, E//2]
        output = out.squeeze(0) # output [N, E//2]
        output = torch.unsqueeze(output, 1) # output --- [N, 1, E//2]
        return output

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class Style_encoder(nn.Module):
    def __init__(self, E, ref_enc_filters, ref_enc_size, ref_enc_strides,
                 ref_enc_pad, ref_enc_gru_size, token_num, num_heads, n_mels):
        super().__init__()
        self.encoder = ReferenceEncoder(E, ref_enc_filters, n_mels)


    def forward(self, inputs):
        enc_out = self.encoder(inputs)

        return enc_out