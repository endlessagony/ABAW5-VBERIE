import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd import Variable

import einops
from   einops import rearrange, repeat
from   einops.layers.torch import Rearrange

from   models.TCN import clones
import math


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def pair(t: int = None):
    return t if isinstance(t, tuple) else (t, t)


def attention(query: torch.Tensor = None, key: torch.Tensor = None, value: torch.Tensor = None, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor = None, key: torch.Tensor = None, value: torch.Tensor = None, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
            for l, x in zip(self.linears, (query, key, value))]

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = None, d_ff: int = None, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor = None):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class MultiModalSublayerConnection(nn.Module):
    def __init__(self, size: int = None, modal_num: int = None, dropout: float = 0.0):
        super(MultiModalSublayerConnection, self).__init__()

        self.modal_num = modal_num
        self.norm = nn.ModuleList()
        
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor = None, sublayer: nn.Module = None):
        residual = x
        _x_list = []
        _x = torch.chunk(x, self.modal_num, -1)

        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))

        x = torch.cat(_x_list, dim=-1)
        return self.dropout(sublayer(x)) + residual
    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features)).to(DEVICE)
        self.b_2 = nn.Parameter(torch.zeros(features)).to(DEVICE)
        self.eps = eps

    def forward(self, x: torch.Tensor = None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    def __init__(self, size: int = None, dropout: float = None):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor = None, sublayer: nn.Module = None):
        return x + self.dropout(sublayer(self.norm(x)))


class PreNorm(nn.Module):
    def __init__(self, dim: int = None, fn: nn.Module = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor = None, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int = None, hidden_dim: int = None, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),    
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor = None):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int = None, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor = None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int = None, depth: int = 4, heads: int = 6, dim_head: int = 128, mlp_dim: int = 126, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x: torch.Tensor = None):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class MultiModalEncoder(nn.Module):
    def __init__(self, layer: nn.Module = None, N: int = None, modal_num: int = None):
        super(MultiModalEncoder, self).__init__()

        self.modal_num = modal_num
        self.layers = layer
        self.norm = nn.ModuleList()

        for i in range(self.modal_num):
            self.norm.append(LayerNorm(layer[0].size))

    def forward(self, x: torch.Tensor = None, mask = None):
        for layer in self.layers:
            x = layer(x, mask)

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []

        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))

        x = torch.cat(_x_list, dim=-1)

        return x
    

class MultiModalAttention(nn.Module):
    def __init__(self, h: int = None, d_model: int = None, modal_num: int = None, dropout: float = 0.1):
        super(MultiModalAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.modal_num = modal_num
        self.mm_linears = nn.ModuleList()

        for i in range(self.modal_num):
            linears = clones(nn.Linear(d_model, d_model), 4)
            self.mm_linears.append(linears)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor = None, key: torch.Tensor = None, value: torch.Tensor = None, mask=None):
        query = torch.chunk(query, self.modal_num, dim=-1)
        key = torch.chunk(key, self.modal_num, dim=-1)
        value = torch.chunk(value, self.modal_num, dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query[0].size(0)
        _query_list = []
        _key_list = []
        _value_list = []

        for i in range(self.modal_num):
            _query_list.append(self.mm_linears[i][0](
                query[i]).view(nbatches, -1, self.h, self.d_k))

            _key_list.append(self.mm_linears[i][1](
                key[i]).view(nbatches, -1, self.h, self.d_k))

            _value_list.append(self.mm_linears[i][2](
                value[i]).view(nbatches, -1, self.h, self.d_k))

        mm_query = torch.stack(_query_list, dim=-2)
        mm_key = torch.stack(_key_list, dim=-2)
        mm_value = torch.stack(_value_list, dim=-2)

        x, _ = attention(mm_query, mm_key, mm_value,mask=mask, dropout=self.dropout)
        x = x.transpose(-2, -3).contiguous().view(nbatches, - 1, self.modal_num, self.h * self.d_k)
        _x = torch.chunk(x, self.modal_num, dim=-2)

        _x_list = []

        for i in range(self.modal_num):
            _x_list.append(self.mm_linears[i][-1](_x[i].squeeze()))

        x = torch.cat(_x_list, dim=-1)

        return x
    

class MultiModalEncoderLayer(nn.Module):
    def __init__(self, size: int = None, modal_num: int = None, mm_atten: nn.Module = None, mt_atten: nn.Module = None, feed_forward: nn.Module = None, dropout: float = None):
        super(MultiModalEncoderLayer, self).__init__()

        self.modal_num = modal_num
        self.mm_atten = mm_atten
        self.mt_atten = mt_atten
        self.feed_forward = feed_forward

        mm_sublayer = MultiModalSublayerConnection(size, modal_num, dropout)
        mt_sublayer = nn.ModuleList()

        for i in range(modal_num):
            mt_sublayer.append(SublayerConnection(size, dropout))

        ff_sublayer = nn.ModuleList()

        for i in range(modal_num):
            ff_sublayer.append(SublayerConnection(size, dropout))

        self.sublayer = nn.ModuleList()
        self.sublayer.append(mm_sublayer)
        self.sublayer.append(mt_sublayer)
        self.sublayer.append(ff_sublayer)

        self.size = size

    def forward(self, x: torch.Tensor = None, mask = None):
        x = self.sublayer[0](x, lambda x: self.mm_atten(x, x, x))
        _x = torch.chunk(x, self.modal_num, dim=-1)

        _x_list = []

        for i in range(self.modal_num):
            feature = self.sublayer[1][i](_x[i], lambda x: self.mt_atten[i](x, x, x, mask=None))
            feature = self.sublayer[2][i](feature, self.feed_forward[i])

            _x_list.append(feature)

        x = torch.cat(_x_list, dim=-1)

        return x


class ModalEncoder(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 6, depth: int = 2, dim_head: int = 64, dropout: float = 0.3, mlp_dim: int = 512, hidden_dim: int = 512) -> None: 
        super().__init__() 
        self.att = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.trans = Transformer(dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = dropout)

    def forward(self, x: torch.Tensor = None):
        att = self.att(x)
        trans = self.trans(x)
        x = torch.cat((x, att, trans), 2)   
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = None, dropout: int = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)

        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0).to(DEVICE)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor = None):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)