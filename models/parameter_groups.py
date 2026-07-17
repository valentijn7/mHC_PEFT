# models/parameter_groups.py
""" utilities for classifying trainable PEFT and mHC parameters """

from dataclasses import dataclass


LORA_PARAMETER_MARKERS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",            
)

VERA_PARAMETER_MARKERS = (
    "vera_lambda",
    "vera_lambda_b",
    "vera_lambda_d",
    "vera_A",
    "vera_B",
)

IA3_PARAMETER_MARKERS = (
    "ia3_l",
)

PROMPT_TUNING_PARAMETER_MARKERS = (
    "prompt_encoder",
    "prompt_embeddings",
)

ADAPTER_PARAMETER_MARKERS = (
    *LORA_PARAMETER_MARKERS,
    *VERA_PARAMETER_MARKERS,
    *IA3_PARAMETER_MARKERS,
    *PROMPT_TUNING_PARAMETER_MARKERS,
)


HC_MODULE_MARKERS = (
    "attn_hc",
    "mlp_hc",
    "attn_delta_kromhc",
    "mlp_delta_kromhc",
    "delta_kromhc",
    "adapter_delta_kromhc",
    "path_lora",
    "residual_memory",
)

HC_PARAMETER_MARKERS = (
    "pre_logits",                       # static SHC / mHC
    "post_logits",
    "res_logits",
    "readout_logits",

    "static_alpha",                     # KromHC / dynamic KromHC
    "dynamic_alpha_fn",
    "pre_branch_scale",
    "residual_scale",
    "static_beta",
    "dynamic_beta_fn",
    "h_post_scale",
    ".norm.gamma",

    "route_gamma",                      # delta-routing / for residual memory adapters
    "delta_route_gamma",
    "adapter_route_gamma",
    "memory_gamma",
    "adapter_route_logit",
    "adapter_route_log_rho",
)


def is_adapter_parameter_name(name: str) -> bool:
    """ return 1 for known Hugging Face PEFT adapter parameters """
    return any(marker in name for marker in ADAPTER_PARAMETER_MARKERS)


def is_hc_parameter_name(name: str) -> bool:
    """ return 1 for trainable HC / KromHC / residual-memory parameters

    Important: adapter params can live inside attn_hc/mlp_hc names when PEFT is
    injected after KromHC, so adapter names must be excluded first
    """
    if is_adapter_parameter_name(name):
        return False

    if "readout_logits" in name:
        return True

    if not any(marker in name for marker in HC_MODULE_MARKERS):
        return False

    return any(marker in name for marker in HC_PARAMETER_MARKERS)


@dataclass
class ParameterGroupSplit:
    hc_params: list
    adapter_params: list
    unknown_params: list
    hc_names: list
    adapter_names: list
    unknown_names: list


def split_trainable_hc_adapter_parameters(model, strict: bool = True) -> ParameterGroupSplit:
    """ split trainable parameters into HC and adapter groups

    If strict = True, unknown trainable parameters raise an error. This prevents
    accidentally training frozen backbone weights with the adapter LR
    """
    hc_params = []
    adapter_params = []
    unknown_params = []

    hc_names = []
    adapter_names = []
    unknown_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

                                        # Adapter check must come first because
                                        # LoRA/VeR params may be inside module
                                        # names containing attn_hc or mlp_hc
        if is_adapter_parameter_name(name):
            adapter_params.append(param)
            adapter_names.append(name)
        elif is_hc_parameter_name(name):
            hc_params.append(param)
            hc_names.append(name)
        else:
            unknown_params.append(param)
            unknown_names.append(name)

    if strict and unknown_names:
        preview = "\n".join(f"  - {name}" for name in unknown_names[:30])
        raise ValueError(
            "Found trainable parameters that are neither recognized as adapter "
            "nor HC/KromHC parameters. This is unsafe for dual-LR training.\n"
            f"{preview}"
        )

    return ParameterGroupSplit(
        hc_params = hc_params,
        adapter_params = adapter_params,
        unknown_params = unknown_params,
        hc_names = hc_names,
        adapter_names = adapter_names,
        unknown_names = unknown_names,
    )