# models/dynamic_mHC.py
""" dynamic manifold-constrained hyper-connections """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from random import randrange

class RMSNorm(nn.Module):
    """ root-mean-square normalization layer """
    def __init__(self, dim, eps = 1e-6):
        """ initialize RMSNorm parameters """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """ apply RMSNorm to input tensor """
                                        # rms = root mean square = sqrt(mean(x^2) + eps)
        rms = x.pow(2).mean(dim = -1, keepdim = True).add(self.eps).sqrt()
        return (x / rms) * self.weight  # normalize


def sinkhorn_logspace(logits, num_iters = 20, eps = 1e-6):
    """ compute a doubly-stochastic matrix via Sinkhorn in log-space """
    z = logits.float()                  # logits: (B,T,n,n)
                                        # numerical stabilization, prevent overflow in exp
    z = z - z.amax(dim = (-2, -1), keepdim = True)

    for _ in range(num_iters):          # normalize by iteration in log space
                                        # row normalize in log-space
        z = z - torch.logsumexp(z, dim = -1, keepdim = True)
                                        # col normalize in log-space
        z = z - torch.logsumexp(z, dim = -2, keepdim = True)
    p = z.exp()                         # exponaniate back to probability space

                                        # final clean normalize 
    p = p / p.sum(dim = -1, keepdim = True).clamp_min(eps)
    p = p / p.sum(dim = -2, keepdim = True).clamp_min(eps)
    return p.to(logits.dtype)


class MHC(nn.Module):
    """ mHC wrapper based on the DeepSeek mHC paper """
    def __init__(
        self,
        branch: nn.Module,
        hidden_size: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        init_std: float = 1e-3,
        train_branch: bool = False,
        layer_index: int | None = None, 
    ):
        """ initialize the MHC wrapper and projection parameters """
        super().__init__()
        self.branch = branch 
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.init_std = init_std        # small init std to break symmetry
        self.init_idx = randrange(num_streams) if layer_index is None else layer_index % num_streams

        flat_dim = num_streams * hidden_size

        self.norm = RMSNorm(flat_dim, eps = eps)

                                        # projection matrix for pre, post and res
        self.pre_proj = nn.Linear(flat_dim, num_streams)
        self.post_proj = nn.Linear(flat_dim, num_streams)
        self.res_proj = nn.Linear(flat_dim, num_streams * num_streams)

                                        # alpha for pre, post and res
        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

        if not train_branch:
            for p in self.branch.parameters():
                p.requires_grad = False
        
                                        # for diagnostics() getter function
        self.diagnostics_enabled = False
        self._last_h_pre = None         
        self._last_h_res = None         
        self._last_h_post = None        

        self.reset_parameters()

    def reset_parameters(self):
        """ initialize projection weights and gating parameters """
                                        # zero-init the dynamic projection weights so each
                                        # gate equals exactly its static bias value at init
        nn.init.zeros_(self.pre_proj.weight)
        nn.init.zeros_(self.post_proj.weight)
        nn.init.zeros_(self.res_proj.weight)

                                        # H_pre: one-hot symmetry-breaking (KromHC / mHC-lite style)
        pre_sel = 8.0
        sig_sel = 1.0 / (1.0 + math.exp(-pre_sel))
        if self.num_streams > 1:
            p_off   = (1.0 - sig_sel) / (self.num_streams - 1)
            pre_off = math.log(p_off / (1.0 - p_off))
        else:
            pre_off = pre_sel
        nn.init.constant_(self.pre_proj.bias, pre_off)
        self.pre_proj.bias.data[self.init_idx] = pre_sel

                                        # H_post[i] = 2 * sigmoid(0) = 1 so that branch
                                        # output is initially unscaled
        nn.init.zeros_(self.post_proj.bias)

                                        # diagonal-dominant bias for res_proj so the doubly-stochastic
                                        # matrix starts approximately identity
        nn.init.zeros_(self.res_proj.bias)
        for i in range(self.num_streams):
            self.res_proj.bias.data[i * self.num_streams + i] = 8.0

                                        # small scale so dynamic part starts near zero
        nn.init.constant_(self.alpha_pre,  1e-2)
        nn.init.constant_(self.alpha_post, 1e-2)
        nn.init.constant_(self.alpha_res,  1e-2)

        nn.init.ones_(self.norm.weight) # RMSNorm scale starts neutral

    @staticmethod
    def _extract_hidden(out):
        """ extract hidden states from a model output tuple """
        return out[0] if isinstance(out, tuple) else out

    def enable_diagnostics(self, enabled = True):
        """ enable or disable caching of routing objects for diagnostics """
        self.diagnostics_enabled = enabled
        if not enabled:                 
            self._last_h_pre = None
            self._last_h_res = None
            self._last_h_post = None

    def diagnostics(self):
        """ return latest dynamic routing objects for generic diagnostics """
        if (
            self._last_h_pre is None
            or self._last_h_res is None
            or self._last_h_post is None
        ):
            raise RuntimeError(
                "no cached routing state found; run a forward pass before diagnostics()"
            )
        return {
            "num_streams": self.num_streams,
            "h_pre": self._last_h_pre,
            "h_res": self._last_h_res,
            "h_post": self._last_h_post,
        }

    def forward(self, X, *args, **kwargs):
        """ run the wrapped branch and update hyperconnection streams """
        B, T, n, D = X.shape            # X: (B,T,n,D)

        flat = X.reshape(B, T, n * D)   # flatten streams into feature vector, normalize
        h = self.norm(flat)
                                        
        h_pre = torch.sigmoid(          # add bias after multiplication with alpha to follow the paper
            self.alpha_pre * F.linear(h, self.pre_proj.weight, bias = None) + self.pre_proj.bias
        )
        h_post = 2.0 * torch.sigmoid(
            self.alpha_post * F.linear(h, self.post_proj.weight, bias = None) + self.post_proj.bias
        )

                                        # calculate res and make doubly stochastic via Sinkhorn
        res_logits = self.alpha_res * F.linear(h, self.res_proj.weight, bias = None) + self.res_proj.bias
        res_logits = res_logits.view(B, T, n, n)
        h_res = sinkhorn_logspace(res_logits, num_iters = self.sinkhorn_iters, eps = self.eps)

        h_pre = h_pre.to(dtype = X.dtype)
        h_post = h_post.to(dtype = X.dtype)
        h_res = h_res.to(dtype = X.dtype)

        if self.diagnostics_enabled:    # for diagnostics() getter function
            self._last_h_pre = h_pre.detach()
            self._last_h_res = h_res.detach()
            self._last_h_post = h_post.detach()

                                        # wrap the neural sub layer; (B,T,D)
        branch_in = torch.sum(h_pre.unsqueeze(-1) * X, dim = 2)              
        branch_in = branch_in.to(dtype = X.dtype)
        branch_out_raw = self.branch(branch_in, *args, **kwargs)
        branch_out = self._extract_hidden(branch_out_raw)

                                        # branch output into streams; (B,T,n,D)
        X_res = torch.einsum("btij,btjd->btid", h_res, X)     
                                        # (B,T,n,D)              
        X_post = h_post.unsqueeze(-1) * branch_out.unsqueeze(2)

        X_new = X_res + X_post          # hyperconnection update
        X_new = X_new.to(dtype = X.dtype)

                                        # return the full stream tensor; readout to (B,T,D)
                                        # happens once at the model level, not after every layer
        if isinstance(branch_out_raw, tuple):
            return (X_new,) + branch_out_raw[1:]
        return X_new