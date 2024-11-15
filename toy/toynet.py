# Borrowing this implentation from https://github.com/bot66/MNISTDiffusion
# 
# Added the following modifications:
# - Using Positional Embeddings instead of learned embeddings (Which only work in discrete space)
# - Using AdaGN with PixelNorm like sCM recommends to inject time embeddings
# - Added logvar output
# - Using GroupNorm instead of BatchNorm: I found BatchNorm does not play nicely with JVP
#
# MIT License
# Copyright (c) 2022 Guocheng Tan
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        half_dim = num_channels // 2
        emb = math.log(10) / (half_dim - 1)
        self.register_buffer('freqs', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, x):
        y = x.to(torch.float32)
        y = y.outer(self.freqs.to(torch.float32))
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=1) * np.sqrt(2)
        return y.to(x.dtype)
    
class Net(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=256):
        super().__init__()
        self.time_embedding = PositionalEmbedding(embedding_dim)
        
        self.pred_net = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.logvar_net = nn.Sequential(
            nn.Linear(embedding_dim, 1, bias=False)
        )
        
    def forward(self, x, t, return_logvar=False):
        # Embed both x and t using positional embeddings
        t_emb = self.time_embedding(t)
        
        # Concatenate embeddings
        h = torch.cat([x.unsqueeze(-1), t_emb], dim=-1)
        
        # Pass through separate networks
        preds = self.pred_net(h).squeeze(-1)
        
        if return_logvar:
            logvar = self.logvar_net(t_emb).squeeze(-1)
            return preds, logvar
        else:
            return preds
        
    def norm_logvar(self):
        eps = 1e-8
        for param in self.logvar_net.parameters():
            param.data.copy_(param.data / torch.linalg.vector_norm(param.data, keepdim=True))



if __name__=="__main__":
    x=torch.randn(1)
    t=torch.tensor([1.1], dtype=torch.float32)
    model=Net(embedding_dim=256, hidden_dim=256)
    y=model(x,t)
    print(y)