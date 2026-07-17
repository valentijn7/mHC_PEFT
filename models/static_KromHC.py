# models/static_KromHC.py
""" Static KromHC implementation """

from __future__ import annotations
from typing import Callable
from functools import partial
from random import randrange
import math

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from models.KromHC import (
    exists,
    default,
    divisible_by,
    add,
    Residual,
    factorize_into_twos,
    get_all_permutations,
    get_cached_2x2_perms,
    get_expand_reduce_stream_functions,
    perm_mats_general,
)


class StaticKromHC(Module):
    """ static Kronecker low-rank hyper-connections """

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: Module | None = None,
        add_branch_out_to_residual = True,
        num_input_views = 1,
        depth_residual_fn = add,
        num_fracs = 1,
        ablate_mapping = None,
        hres_init = "identity",
        hres_init_noise_std = 0.0,
        drop_hres_when_ablated = True,
    ):
        super().__init__()

        if ablate_mapping is None:
            ablate_mapping = []
        elif isinstance(ablate_mapping, str):
            ablate_mapping = [ablate_mapping]

        ablate_set = {str(m).lower() for m in ablate_mapping}
        unknown = ablate_set - {"pre", "post", "res"}
        assert not unknown, (
            f"unknown ablation target(s) {unknown}; "
            "expected any of 'pre', 'post', 'res'"
        )

        self.ablate_pre = "pre" in ablate_set
        self.ablate_post = "post" in ablate_set
        self.ablate_res = "res" in ablate_set

        self.drop_hres = bool(self.ablate_res and drop_hres_when_ablated)

        assert hres_init in ("identity", "uniform", "permutation"), (
            f"unknown hres_init {hres_init!r}; "
            "expected 'identity', 'uniform', or 'permutation'"
        )
        assert hres_init_noise_std >= 0.0, (
            "hres_init_noise_std must be non-negative"
        )

        self.hres_init = hres_init
        self.hres_init_noise_std = float(hres_init_noise_std)

        self.branch = branch

        assert num_fracs == 1, (
            "StaticKromHC currently supports num_fracs == 1. "
            "This matches the current KromHC experimental setting."
        )
        assert num_input_views == 1, (
            "StaticKromHC currently supports num_input_views == 1."
        )

        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange(
            'b ... (f d) -> b ... f d',
            f = num_fracs,
        )
        self.merge_fracs = Rearrange(
            'b ... f d -> b ... (f d)'
        )

        assert divisible_by(dim, num_fracs), (
            f"feature dimension ({dim}) must be divisible by "
            f"the num_fracs ({num_fracs})"
        )

        dim //= num_fracs

        assert num_residual_streams >= 2, (
            "num_residual_streams must be at least 2"
        )
        assert num_residual_streams & (num_residual_streams - 1) == 0, (
            f"num_residual_streams must be a power of 2, "
            f"got {num_residual_streams}"
        )

        self.num_residual_streams = num_residual_streams
        self.num_streams = num_residual_streams

        init_residual_index = (
            default(layer_index, randrange(num_residual_streams))
            % num_residual_streams
        )

        self.num_input_views = num_input_views

        self.factors = factorize_into_twos(num_residual_streams)
        self.num_factors = len(self.factors)

        self.factor_perms = []
        total_res_coeffs = 0
        for factor_size in self.factors:
            num_perms = math.factorial(factor_size)
            self.factor_perms.append(num_perms)
            total_res_coeffs += num_perms

        for factor_size in self.factors:
            if factor_size > 2 and (factor_size, "cpu") not in perm_mats_general:
                perm_mats_general[(factor_size, "cpu")] = (
                    get_all_permutations(factor_size).to("cpu")
                )

        res_coeffs = 0 if self.drop_hres else total_res_coeffs

        pre_sel = 8.0
        sig_sel = 1.0 / (1.0 + math.exp(-pre_sel))

        if num_residual_streams > 1:
            p_off = (1.0 - sig_sel) / (num_residual_streams - 1)
            pre_off = math.log(p_off / (1.0 - p_off))
        else:
            pre_off = pre_sel

        init_alpha0 = torch.full(
            (num_residual_streams, num_input_views),
            float(pre_off),
        )
        init_alpha0[init_residual_index, :] = float(pre_sel)

        high, low = 0.0, -12.0

        if self.drop_hres:
            init_alpha1 = torch.empty(0)
        else:
            init_alpha1 = torch.full((total_res_coeffs,), float(low))

            if self.hres_init == "uniform":
                init_alpha1.fill_(0.0)
            elif self.hres_init_noise_std <= 0.0:
                coeff_idx = 0
                for num_perms in self.factor_perms:
                    sel = 0 if self.hres_init == "identity" else (num_perms - 1)
                    init_alpha1[coeff_idx + sel] = float(high)
                    coeff_idx += num_perms
            else:
                coeff_idx = 0
                for num_perms in self.factor_perms:
                    sel = 0 if self.hres_init == "identity" else (num_perms - 1)

                    off = (
                        torch.randn(num_perms)
                        * self.hres_init_noise_std
                    ).abs()
                    off[sel] = 0.0

                    cap = 0.9 / max(num_perms - 1, 1)
                    off = off.clamp(max = cap)

                    probs = off.clone()
                    probs[sel] = 1.0 - off.sum()
                    probs = probs.clamp_min(math.exp(low))

                    init_alpha1[coeff_idx:coeff_idx + num_perms] = torch.log(
                        probs
                    )
                    coeff_idx += num_perms

        self.static_alpha = nn.Parameter(
            cat(
                [
                    init_alpha0.view(-1),
                    init_alpha1,
                ],
                dim = -1,
            )
        )

        self.total_res_coeffs = total_res_coeffs
        self.res_coeffs = res_coeffs

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            beta_init = torch.zeros(num_residual_streams)
            self.static_beta = nn.Parameter(beta_init)

        self.dropout = nn.Dropout(dropout)

        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

        self.diagnostics_enabled = False
        self._last_h_pre = None
        self._last_h_res = None
        self._last_h_post = None

    def _get_factor_perms(self, factor_size, device):
        """get permutation matrices for a given factor size"""
        if factor_size == 2:
            return get_cached_2x2_perms(device)

        dev_key = str(device)
        if (factor_size, dev_key) not in perm_mats_general:
            perm_mats_general[(factor_size, dev_key)] = (
                get_all_permutations(factor_size).to(device)
            )
        return perm_mats_general[(factor_size, dev_key)]

    def _build_static_kronecker_hres(self, static_coeffs, device, batch_shape):
        """build static H_res from Kronecker product factor coefficients"""
        if len(self.factors) == 0:
            eye = static_coeffs.new_ones(1, 1)
            return eye.expand(*batch_shape, 1, 1)

        all_2x2 = all(factor_size == 2 for factor_size in self.factors)

        if all_2x2:
            coeffs = static_coeffs.view(self.num_factors, 2)
            weights = F.softmax(coeffs, dim = -1)

            p = weights[:, 0]
            one_minus_p = 1.0 - p

            row0 = torch.stack([p, one_minus_p], dim = -1)
            row1 = torch.stack([one_minus_p, p], dim = -1)
            factor_matrices = torch.stack([row0, row1], dim = -2)

            result = factor_matrices[0]
            for factor_index in range(1, self.num_factors):
                mat = factor_matrices[factor_index]

                result_exp = rearrange(result, 'a b -> a 1 b 1')
                mat_exp = rearrange(mat, 'c d -> 1 c 1 d')

                kron = result_exp * mat_exp
                result = kron.reshape(
                    result.shape[-2] * 2,
                    result.shape[-1] * 2,
                )

            return result.expand(*batch_shape, *result.shape)

        factor_matrices = []
        coeff_idx = 0

        for factor_size, num_perms in zip(self.factors, self.factor_perms):
            factor_coeffs = static_coeffs[coeff_idx:coeff_idx + num_perms]
            coeff_idx += num_perms

            perms = self._get_factor_perms(factor_size, device)
            weights = F.softmax(factor_coeffs, dim = -1)
            factor_matrix = torch.einsum("r,rij->ij", weights, perms)
            factor_matrices.append(factor_matrix)

        result = factor_matrices[0]
        for mat in factor_matrices[1:]:
            result_exp = rearrange(result, 'a1 a2 -> a1 1 a2 1')
            mat_exp = rearrange(mat, 'b1 b2 -> 1 b1 1 b2')
            kron = result_exp * mat_exp
            result = rearrange(kron, 'a b c d -> (a b) (c d)')

        return result.expand(*batch_shape, *result.shape)

    def enable_diagnostics(self, enabled = True):
        """enable or disable caching of routing objects for diagnostics"""
        self.diagnostics_enabled = enabled

        if not enabled:
            self._last_h_pre = None
            self._last_h_res = None
            self._last_h_post = None

    def diagnostics(self):
        """return latest static KromHC routing objects"""
        if (
            self._last_h_pre is None
            or self._last_h_res is None
            or self._last_h_post is None
        ):
            raise RuntimeError(
                "no cached routing state found; call enable_diagnostics(True) "
                "and run a forward pass before diagnostics()"
            )

        return {
            "num_streams": self.num_residual_streams,
            "h_pre": self._last_h_pre,
            "h_res": self._last_h_res,
            "h_post": self._last_h_post,
        }

    def width_connection(self, residuals):
        """read from residual streams using static routing only"""
        streams = self.num_residual_streams

        _ = self.residual_transform(residuals)

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        residuals = self.split_fracs(residuals)
        residuals = rearrange(
            residuals,
            '(b s) ... d -> b ... s d',
            s = streams,
        )

        batch_shape = residuals.shape[:-2]
        device = residuals.device

        psize = self.num_input_views * streams

        static_pre = self.static_alpha[:psize]
        alpha_pre = rearrange(
            static_pre,
            '(s v) -> s v',
            s = streams,
            v = self.num_input_views,
        )
        alpha_pre = alpha_pre.sigmoid()

        if self.ablate_pre:
            alpha_pre = torch.full_like(
                alpha_pre,
                1.0 / self.num_residual_streams,
            )

        alpha_pre = alpha_pre.view(
            *((1,) * len(batch_shape)),
            streams,
            1,
            self.num_input_views,
        )
        alpha_pre = alpha_pre.expand(
            *batch_shape,
            streams,
            1,
            self.num_input_views,
        )

        if self.ablate_res:
            eye = torch.eye(
                streams,
                device = device,
                dtype = self.static_alpha.dtype,
            )
            alpha_residual = eye.expand(*batch_shape, streams, streams)
        else:
            static_residual = self.static_alpha[psize:]
            alpha_residual = self._build_static_kronecker_hres(
                static_coeffs = static_residual,
                device = device,
                batch_shape = batch_shape,
            )

        alpha_residual = alpha_residual.unsqueeze(-2)

        alpha = cat((alpha_pre, alpha_residual), dim = -1)

        beta = None
        if self.add_branch_out_to_residual:
            beta = rearrange(
                self.static_beta,
                '(s f) -> s f',
                s = streams,
                f = self.num_fracs,
            )
            beta = beta.sigmoid() * 2.0

            if self.ablate_post:
                beta = torch.ones_like(beta)

            beta = beta.view(
                *((1,) * len(batch_shape)),
                streams,
                self.num_fracs,
            )
            beta = beta.expand(*batch_shape, streams, self.num_fracs)

        if self.diagnostics_enabled:
            if self.num_fracs != 1:
                raise RuntimeError(
                    "diagnostics for StaticKromHC expects num_fracs == 1"
                )

            if self.num_input_views != 1:
                raise RuntimeError(
                    "diagnostics for StaticKromHC expects num_input_views == 1"
                )

            h_pre_diag = alpha_pre.squeeze(-1).squeeze(-1)
            h_res_diag = alpha_residual.squeeze(-2).squeeze(-2)

            if beta is None:
                h_post_diag = torch.ones_like(h_pre_diag)
            else:
                h_post_diag = beta.squeeze(-1)

            self._last_h_pre = h_pre_diag.detach()
            self._last_h_res = h_res_diag.detach()
            self._last_h_post = h_post_diag.detach()

        alpha = alpha.to(residuals.dtype)

        if beta is not None:
            beta = beta.to(residuals.dtype)

        mix_h = einsum(
            alpha,
            residuals,
            '... f1 s f2 t, ... f1 s d -> ... f2 t d',
        )

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input = mix_h[..., :self.num_input_views, :]
            residuals = mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        residuals = self.merge_fracs(residuals)

        return branch_input, residuals, dict(beta = beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta,
    ):
        """write branch output to residual streams using static H_post"""
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(
            branch_output,
            beta,
            'b ... f1 d, b ... f1 s f2 -> b ... f2 s d',
        )

        output = rearrange(output, 'b ... s d -> (b s) ... d')
        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        """decorate an external branch with static KromHC routing"""
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        """run static KromHC branch wrapper"""
        branch_input, residuals, residual_kwargs = self.width_connection(
            residuals
        )

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(
                branch_out,
                residuals,
                **residual_kwargs,
            )
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(
            branch_input,
            *branch_args,
            **branch_kwargs,
        )

        return add_residual_fn(branch_output)


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    **kwargs,
):
    """return init function and expand/reduce stream functions for StaticKromHC"""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = StaticKromHC if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        num_fracs = num_fracs,
        **kwargs,
    )

    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams,
        add_stream_embed = add_stream_embed,
        dim = dim,
        disable = disable,
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


StaticKromHC.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
StaticKromHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)