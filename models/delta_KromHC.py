# models/delta_KromHC.py
""" KromHC dynamically applied to adapter delta, for OLMo-2 """

from __future__ import annotations
from contextlib import contextmanager

import math
import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2Model,
    Olmo2DecoderLayer,
)

from models.KromHC import KromHC
from models.static_KromHC import StaticKromHC


@contextmanager
def adapters_disabled(module: nn.Module):
    """ temporarily disables PEFT adapters inside one wrapped branch

    This is intentionally local to the branch. It is used to compute
    base_out = f(x) and adapted_out = f(x) + delta_adapter(x), so the adapter
    delta can be extracted as adapted_out - base_out
    """
    changed_modules = []

    for submodule in module.modules():
        enable_adapters = getattr(submodule, "enable_adapters", None)
        if not callable(enable_adapters):
            continue

        old_disable_adapters = getattr(submodule, "_disable_adapters", None)
        changed_modules.append((submodule, old_disable_adapters))
        enable_adapters(False)

    try:
        yield
    finally:
        for submodule, old_disable_adapters in reversed(changed_modules):
            enable_adapters = getattr(submodule, "enable_adapters", None)
            if not callable(enable_adapters):
                continue

            if old_disable_adapters is None:
                enable_adapters(True)
            else:
                enable_adapters(not bool(old_disable_adapters))


def _logit_from_probability(p: float) -> float:
    """ return logit(p) with light clipping """
    eps = 1e-6
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class AdapterDeltaProvider(nn.Module):
    """ computes a local adapter delta for a branch module

    The branch is called twice:
      adapted_out = branch(x) with adapters enabled
      base_out    = branch(x) with adapters disabled

    The delta is computed on the first tensor leaf of the branch output. This
    matches the existing KromHC code, which applies routing to the first output
    tensor and preserves auxiliary outputs through the pytree spec
    """

    def __init__(self, branch: nn.Module):
        super().__init__()
        self.branch = branch

    @staticmethod
    def _first_tensor(output):
        leaves, _ = tree_flatten(output)
        if not leaves:
            raise RuntimeError("branch output has no leaves")

        first = leaves[0]
        if not torch.is_tensor(first):
            raise RuntimeError(
                "first branch-output leaf must be a tensor, "
                f"got {type(first)}"
            )
        return first

    def forward(self, hidden_states, *branch_args, **branch_kwargs):
        adapted_out = self.branch(hidden_states, *branch_args, **branch_kwargs)
        adapted_tensor = self._first_tensor(adapted_out)

        with adapters_disabled(self.branch):
            base_out = self.branch(hidden_states, *branch_args, **branch_kwargs)
        base_tensor = self._first_tensor(base_out)

        if adapted_tensor.shape != base_tensor.shape:
            raise RuntimeError(
                "adapted and base branch outputs have different shapes: "
                f"{tuple(adapted_tensor.shape)} vs {tuple(base_tensor.shape)}"
            )

        adapter_delta = adapted_tensor - base_tensor
        return base_out, adapted_out, adapter_delta


class _InternalKromHCRouter(KromHC):
    """ KromHC router used inside DeltaKromHC

    diagnostics is intentionally hidden here so generic diagnostics report the
    outer DeltaKromHC module once, including adapter_route_gamma
    """
    diagnostics = None


class _InternalStaticKromHCRouter(StaticKromHC):
    """ Static KromHC router used inside DeltaKromHC

    diagnostics is intentionally hidden here so generic diagnostics report the
    outer DeltaKromHC module once, including adapter_route_gamma
    """
    diagnostics = None


class DeltaKromHC(nn.Module):
    """ KromHC routing applied only to a PEFT adapter delta """
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: nn.Module,
        layer_index: int | None = None,
        num_fracs: int = 1,
        ablate_mapping = None,
        hres_init: str = "identity",
        hres_init_noise_std: float = 0.0,
        gamma_init: float = 1e-3,
        train_gamma: bool = True,
        static_routing: bool = False,
        route_mode: str = "additive",
        write_only: bool = False,
        lambda_init: float = 0.1,
        rho_init: float = 1.0,
        post_init_offset: float = 0.0,
    ):
        super().__init__()

        if ablate_mapping is None:
            ablate_mapping = ["res"]
        elif isinstance(ablate_mapping, str):
            ablate_mapping = [ablate_mapping]
        else:
            ablate_mapping = list(ablate_mapping)

        self.num_residual_streams = num_residual_streams
        self.num_streams = num_residual_streams
        self.branch = branch
        self.delta_provider = AdapterDeltaProvider(branch)

                                        # choose static or dynamic based on bool(static_routing)
        self.static_routing = bool(static_routing)
        self.write_only = bool(write_only)
        self.route_mode = str(route_mode)
        valid_route_modes = {
            "additive",
            "residualized_affine",
            "sigmoid",
            "sigmoid_rho",
        }
        if self.route_mode not in valid_route_modes:
            raise ValueError(
                f"unknown delta-KromHC route_mode={self.route_mode!r}; "
                f"expected one of {sorted(valid_route_modes)}"
            )
        router_cls = (                  
            _InternalStaticKromHCRouter
            if self.static_routing
            else _InternalKromHCRouter
        )
        self.router = router_cls(
            num_residual_streams,
            dim = dim,
            branch = None,
            layer_index = layer_index,
            num_fracs = num_fracs,
            ablate_mapping = ablate_mapping,
            hres_init = hres_init,
            hres_init_noise_std = hres_init_noise_std,
        )

        self.post_init_offset = float(post_init_offset)

        if self.post_init_offset < 0.0:
            raise ValueError("post_init_offset must be non-negative")
        if self.post_init_offset > 0.0 and not self.write_only:
            raise ValueError(
                "delta_kromhc_post_init_offset is intended only for write_only=True"
            )
        if self.write_only and self.post_init_offset > 0.0:
            self._break_post_symmetry(self.post_init_offset)

        self.adapter_route_gamma = nn.Parameter(
            torch.tensor(float(gamma_init), dtype = torch.float32)
        )
        self.adapter_route_gamma.requires_grad_(bool(train_gamma))

        self.adapter_route_logit = nn.Parameter(
            torch.tensor(
                _logit_from_probability(lambda_init),
                dtype = torch.float32,
            )
        )
        self.adapter_route_logit.requires_grad_(bool(train_gamma))

        if float(rho_init) <= 0.0:
            raise ValueError("rho_init must be positive")
        self.adapter_route_log_rho = nn.Parameter(
            torch.tensor(
                math.log(float(rho_init)),
                dtype = torch.float32,
            )
        )
        self.adapter_route_log_rho.requires_grad_(bool(train_gamma))

        self.diagnostics_enabled = False
        self._last_adapter_delta_norm = None
        self._last_plain_delta_norm = None
        self._last_routed_delta_norm = None
        self._last_effective_delta_norm = None
        self._last_gamma = None
        self._last_lambda = None
        self._last_rho = None

    def _break_post_symmetry(self, magnitude: float):
        """deterministically break H_post symmetry for write-only routing"""
        static_beta = getattr(self.router, "static_beta", None)
        if static_beta is None:
            raise RuntimeError(
                "post symmetry breaking requested, but router has no static_beta"
            )

        with torch.no_grad():
            pattern = torch.linspace(
                -1.0,
                1.0,
                steps = static_beta.numel(),
                device = static_beta.device,
                dtype = static_beta.dtype,
            )
            pattern = pattern - pattern.mean()
            pattern = pattern / pattern.abs().max().clamp_min(1e-12)
            pattern = pattern.reshape_as(static_beta)

            static_beta.add_(float(magnitude) * pattern)

    def enable_diagnostics(self, enabled = True):
        """ enable or disable routing-state caching """
        self.diagnostics_enabled = enabled
        self.router.enable_diagnostics(enabled)

        if not enabled:
            self._last_adapter_delta_norm = None
            self._last_plain_delta_norm = None
            self._last_routed_delta_norm = None
            self._last_effective_delta_norm = None
            self._last_gamma = None
            self._last_lambda = None
            self._last_rho = None

    def diagnostics(self):
        """ return latest routing objects plus gamma for diagnostics """
        if (
            self.router._last_h_pre is None
            or self.router._last_h_res is None
            or self.router._last_h_post is None
        ):
            raise RuntimeError(
                "no cached routing state found; call enable_diagnostics(True) "
                "and run a forward pass before diagnostics()"
            )

        gamma = self.adapter_route_gamma.detach().float().cpu()

        return {
            "num_streams": self.num_residual_streams,
            "h_pre": self.router._last_h_pre,
            "h_res": self.router._last_h_res,
            "h_post": self.router._last_h_post,

            "route_mode": self.route_mode,
            "write_only": self.write_only,
            "post_init_offset": self.post_init_offset,

            "adapter_route_gamma": gamma,
            "adapter_route_gamma_abs": gamma.abs(),
            "adapter_route_lambda": self._last_lambda,
            "adapter_route_rho": self._last_rho,

            "adapter_delta_norm": self._last_adapter_delta_norm,
            "plain_delta_norm": self._last_plain_delta_norm,
            "routed_delta_norm": self._last_routed_delta_norm,
            "effective_delta_norm": self._last_effective_delta_norm,
        }

    def _repeat_to_streams(self, tensor, residuals):
        """ repeat a (B,...) tensor to match folded stream residuals (B*S,...) """
        streams = self.num_residual_streams

        if tensor.shape[0] * streams != residuals.shape[0]:
            raise RuntimeError(
                "cannot repeat branch output to folded stream batch: "
                f"branch batch={tensor.shape[0]}, streams={streams}, "
                f"residual batch={residuals.shape[0]}"
            )

        return tensor.repeat_interleave(streams, dim = 0).to(residuals.dtype)

    def _mean_over_streams(self, residuals):
        """fixed uniform read from folded streams: (B*S,...) -> (B,...)"""
        streams = self.num_residual_streams

        if residuals.shape[0] % streams != 0:
            raise RuntimeError(
                "cannot unfold folded stream batch: "
                f"batch={residuals.shape[0]}, streams={streams}"
            )

        batch = residuals.shape[0] // streams
        unfolded = residuals.reshape(batch, streams, *residuals.shape[1:])
        return unfolded.mean(dim = 1)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        """run direct/routed adapter-delta combinations"""
        residuals_before_width = residuals

        routed_branch_input, residuals, residual_kwargs = self.router.width_connection(
            residuals
        )

        if self.write_only:
            branch_input = self._mean_over_streams(residuals_before_width)
        else:
            branch_input = routed_branch_input

        base_out, adapted_out, adapter_delta = self.delta_provider(
            branch_input,
            *branch_args,
            **branch_kwargs,
        )
        (adapted_tensor, *rest), tree_spec = tree_flatten(adapted_out)
        if not torch.is_tensor(adapted_tensor):
            raise RuntimeError(
                "first adapted branch-output leaf must be a tensor, "
                f"got {type(adapted_tensor)}"
            )
        (base_tensor, *_), _ = tree_flatten(base_out)
        if not torch.is_tensor(base_tensor):
            raise RuntimeError(
                "first base branch-output leaf must be a tensor, "
                f"got {type(base_tensor)}"
            )

        base_update = self._repeat_to_streams(base_tensor, residuals)
        plain_delta_update = self._repeat_to_streams(adapter_delta, residuals)

        routed_residuals = self.router.depth_connection(
            adapter_delta,
            residuals,
            **residual_kwargs,
        )
        routed_delta_update = routed_residuals - residuals

        gamma = self.adapter_route_gamma.to(
            device = residuals.device,
            dtype = residuals.dtype,
        )
        route_lambda = torch.sigmoid(
            self.adapter_route_logit.to(
                device = residuals.device,
                dtype = residuals.dtype,
            )
        )
        rho = torch.exp(
            self.adapter_route_log_rho.to(
                device = residuals.device,
                dtype = residuals.dtype,
            )
        )

        if self.route_mode == "additive":
            # eerste idee:
            # frozen + LoRA + gamma * KromHC(LoRA)
            effective_delta = plain_delta_update + gamma * routed_delta_update
        elif self.route_mode == "residualized_affine":
            # residualized / affine:
            # LoRA + gamma * (KromHC(LoRA) - LoRA)
            effective_delta = (
                plain_delta_update
                + gamma * (routed_delta_update - plain_delta_update)
            )
        elif self.route_mode == "sigmoid":
            # convex mixture:
            # (1-lambda) * LoRA + lambda * KromHC(LoRA)
            effective_delta = (
                (1.0 - route_lambda) * plain_delta_update
                + route_lambda * routed_delta_update
            )
        elif self.route_mode == "sigmoid_rho":
            # convex mixture with positive adapter gain:
            # rho * [(1-lambda) * LoRA + lambda * KromHC(LoRA)]
            mixed_delta = (
                (1.0 - route_lambda) * plain_delta_update
                + route_lambda * routed_delta_update
            )
            effective_delta = rho * mixed_delta
        else:
            raise RuntimeError(f"unhandled route_mode={self.route_mode!r}")

        output = residuals + base_update + effective_delta

        if self.diagnostics_enabled:
            self._last_gamma = self.adapter_route_gamma.detach().float().cpu()
            self._last_lambda = route_lambda.detach().float().cpu()
            self._last_rho = rho.detach().float().cpu()
            self._last_adapter_delta_norm = (
                adapter_delta.detach().float().norm().cpu()
            )
            self._last_plain_delta_norm = (
                plain_delta_update.detach().float().norm().cpu()
            )
            self._last_routed_delta_norm = (
                routed_delta_update.detach().float().norm().cpu()
            )
            self._last_effective_delta_norm = (
                effective_delta.detach().float().norm().cpu()
            )

        return tree_unflatten((output, *rest), tree_spec)


class _OlmoAttentionBranch(nn.Module):
    """ pre-norm attention branch used inside delta-routed KromHC """

    def __init__(self, layer: Olmo2DecoderLayer):
        super().__init__()
        self.self_attn = layer.self_attn
        self.post_norm = layer.post_attention_layernorm

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache = False,
        position_embeddings = None,
        **kwargs,
    ):
        attn_out, *rest = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            use_cache = use_cache,
            position_embeddings = position_embeddings,
            **kwargs,
        )
        return (self.post_norm(attn_out), *rest)


class _OlmoMLPBranch(nn.Module):
    """ pre-norm MLP branch used inside delta-routed KromHC """

    def __init__(self, layer: Olmo2DecoderLayer):
        super().__init__()
        self.mlp = layer.mlp
        self.post_norm = layer.post_feedforward_layernorm

    def forward(self, hidden_states, **kwargs):
        return self.post_norm(self.mlp(hidden_states))


class DeltaKromHCOlmoDecoderLayer(nn.Module):
    """ OLMo decoder layer with KromHC applied to adapter deltas """

    def __init__(
        self,
        base_layer: Olmo2DecoderLayer,
        hidden_size: int,
        num_streams: int = 2,
        num_fracs: int = 1,
        layer_index: int = 0,
        ablate_mapping = None,
        hres_init: str = "identity",
        hres_init_noise_std: float = 0.0,
        gamma_init: float = 1e-3,
        train_gamma: bool = True,
        static_routing: bool = False,
        route_mode: str = "additive",
        write_only: bool = False,
        lambda_init: float = 0.1,
        rho_init: float = 1.0,
        post_init_offset: float = 0.0,
    ):
        super().__init__()

        self.attn_delta_kromhc = DeltaKromHC(
            num_streams,
            dim = hidden_size,
            branch = _OlmoAttentionBranch(base_layer),
            layer_index = layer_index,
            num_fracs = num_fracs,
            ablate_mapping = ablate_mapping,
            hres_init = hres_init,
            hres_init_noise_std = hres_init_noise_std,
            gamma_init = gamma_init,
            train_gamma = train_gamma,
            static_routing = static_routing,
            route_mode = route_mode,
            write_only = write_only,
            lambda_init = lambda_init,
            rho_init = rho_init,
            post_init_offset = post_init_offset,
        )
        self.mlp_delta_kromhc = DeltaKromHC(
            num_streams,
            dim = hidden_size,
            branch = _OlmoMLPBranch(base_layer),
            layer_index = layer_index,
            num_fracs = num_fracs,
            ablate_mapping = ablate_mapping,
            hres_init = hres_init,
            hres_init_noise_std = hres_init_noise_std,
            gamma_init = gamma_init,
            train_gamma = train_gamma,
            static_routing = static_routing,
            route_mode = route_mode,
            write_only = write_only,
            lambda_init = lambda_init,
            rho_init = rho_init,
            post_init_offset = post_init_offset,
        )

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        use_cache = False,
        position_embeddings = None,
        **kwargs,
    ):
        hidden_states = self.attn_delta_kromhc(
            hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            use_cache = use_cache,
            position_embeddings = position_embeddings,
            **kwargs,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        hidden_states = self.mlp_delta_kromhc(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        return hidden_states


class DeltaKromHCOlmoModel(Olmo2Model):
    """ OLMo model with adapter-delta KromHC streams folded into batch dim """

    def __init__(
        self,
        base_model,
        hidden_size,
        num_streams = 2,
        num_fracs = 1,
        ablate_mapping = None,
        hres_init = "identity",
        hres_init_noise_std = 0.0,
        gamma_init = 1e-3,
        train_gamma = True,
        static_routing = False,
        route_mode = "additive",
        write_only = False,
        lambda_init = 0.1,
        rho_init = 1.0,
        post_init_offset = 0.0,
    ):
        super().__init__(base_model.config)
        self.num_streams = num_streams
        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.embed_tokens = base_model.embed_tokens
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb

        self.expand_stream, self.reduce_stream = KromHC.get_expand_reduce_stream_functions(
            num_streams = num_streams,
        )

        self.layers = nn.ModuleList([
            DeltaKromHCOlmoDecoderLayer(
                base_layer = layer,
                hidden_size = hidden_size,
                num_streams = num_streams,
                num_fracs = num_fracs,
                layer_index = idx,
                ablate_mapping = ablate_mapping,
                hres_init = hres_init,
                hres_init_noise_std = hres_init_noise_std,
                gamma_init = gamma_init,
                train_gamma = train_gamma,
                static_routing = static_routing,
                route_mode = route_mode,
                write_only = write_only,
                lambda_init = lambda_init,
                rho_init = rho_init,
                post_init_offset = post_init_offset,
            )
            for idx, layer in enumerate(base_model.layers)
        ])

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(
                self.layers[0].attn_delta_kromhc.branch.self_attn.q_proj.weight.dtype
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config = self.config)

        if position_ids is None:
            past_seen = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            position_ids = torch.arange(
                inputs_embeds.shape[1],
                device = inputs_embeds.device,
            ) + past_seen
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config = self.config,
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            position_ids = position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids = position_ids)

        hidden_states = self.expand_stream(inputs_embeds)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = causal_mask,
                position_embeddings = position_embeddings,
                position_ids = position_ids,
                past_key_values = past_key_values,
                use_cache = use_cache,
                **kwargs,
            )

        hidden_states = self.reduce_stream(hidden_states)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state = hidden_states,
            past_key_values = past_key_values,
        )


def olmo_delta_kromhc_adapter(
    olmo: nn.Module,
    num_streams: int = 2,
    num_fracs: int = 1,
    ablate_mapping = None,
    hres_init = "identity",
    hres_init_noise_std = 0.0,
    gamma_init = 1e-3,
    train_gamma = True,
    static_routing = False,
    route_mode = "additive",
    write_only = False,
    lambda_init = 0.1,
    rho_init = 1.0,
    post_init_offset = 0.0,
):
    """ replace olmo.model with adapter-delta KromHC model """
    if ablate_mapping is None:
        ablate_mapping = ["res"]
    elif isinstance(ablate_mapping, str):
        ablate_mapping = [ablate_mapping]
    else:
        ablate_mapping = list(ablate_mapping)

    hidden_size = olmo.config.hidden_size
    olmo.model = DeltaKromHCOlmoModel(
        base_model = olmo.model,
        hidden_size = hidden_size,
        num_streams = num_streams,
        num_fracs = num_fracs,
        ablate_mapping = ablate_mapping,
        hres_init = hres_init,
        hres_init_noise_std = hres_init_noise_std,
        gamma_init = gamma_init,
        train_gamma = train_gamma,
        static_routing = static_routing,
        route_mode = route_mode,
        write_only = write_only,
        lambda_init = lambda_init,
        rho_init = rho_init,
        post_init_offset = post_init_offset,
    )
    return olmo