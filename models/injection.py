# models/injection.py
""" inject PEFT methods into model """

from models.baselines import (
    apply_lora,
    apply_ia3,
    apply_dora,
    apply_prompt_tuning,
    apply_layer_tuning,
    apply_vera
)
from models.parameter_groups import is_hc_parameter_name
from models.model_OLMo_2 import (
    olmo_shc,
    olmo_mhc_lite,
    olmo_kromhc,
    olmo_static_kromhc,
    olmo_mhc,
    olmo_static_mhc_lite
)
from models.delta_KromHC import olmo_delta_kromhc_adapter




def freeze_base_model(model):
    """ freeze all existing model params """
    for parameter in model.parameters():
        parameter.requires_grad = False # freeze pretrained weights!


def enable_kromhc_routing_parameters(model):
    """ re-enable only HC/KromHC routing params after PEFT freezes the base model """
    for name, param in model.named_parameters():
        if is_hc_parameter_name(name):
            param.requires_grad = True


def enable_delta_kromhc_parameters(
    model,
    static_routing = False,
    route_mode = "additive",
    train_route = True,
    write_only = False,
):
    """ re-enable delta-KromHC params after PEFT freezes the base model """

    dynamic_markers = (
        "dynamic_alpha_fn",
        "dynamic_beta_fn",
        "pre_branch_scale",
        "h_post_scale",
        ".norm.gamma",
    )

    pre_route_markers = (
        "static_alpha",
        "dynamic_alpha_fn",
        "pre_branch_scale",
    )

    all_route_param_markers = (
        "adapter_route_gamma",
        "adapter_route_logit",
        "adapter_route_log_rho",
    )

    if route_mode in ("additive", "residualized_affine"):
        active_route_param_markers = ("adapter_route_gamma",)
    elif route_mode == "sigmoid":
        active_route_param_markers = ("adapter_route_logit",)
    elif route_mode == "sigmoid_rho":
        active_route_param_markers = (
            "adapter_route_logit",
            "adapter_route_log_rho",
        )
    else:
        raise ValueError(f"unknown delta_kromhc_route_mode={route_mode!r}")

    for name, param in model.named_parameters():

        if any(marker in name for marker in all_route_param_markers):
            param.requires_grad = (
                bool(train_route)
                and any(marker in name for marker in active_route_param_markers)
            )
            continue

        if not is_hc_parameter_name(name):
            continue

        if write_only and any(marker in name for marker in pre_route_markers):
            param.requires_grad = False
            continue

        if static_routing and any(marker in name for marker in dynamic_markers):
            param.requires_grad = False
            continue

        param.requires_grad = True


def inject_hyperconnections(model, method_cfg):
    """ replaces OLMo layers with SHC-wrapped layers """
    freeze_base_model(model)             # train only new shc params
    return olmo_shc(
        model,
        num_streams = method_cfg.shc_num_streams,
        sinkhorn_iters = method_cfg.shc_sinkhorn_iterations,
        eps = method_cfg.shc_sinkhorn_epsilon,
        train_branch = method_cfg.shc_train_wrapped_branch,
        dropout_res=method_cfg.shc_dropout_res,
        ablate_mapping = method_cfg.shc_ablation_mapping,
        noise_std=method_cfg.shc_noise_std,
        dropout_stream = method_cfg.shc_dropout_stream,
        softmax_readout = getattr(method_cfg, "shc_softmax_readout", False),
    )


def inject_mhc(model, method_cfg):
    device = next(model.parameters()).device
    freeze_base_model(model)
    return olmo_mhc(
        model,
        num_streams=method_cfg.shc_num_streams,
        sinkhorn_iters=method_cfg.shc_sinkhorn_iterations,
        eps=method_cfg.shc_sinkhorn_epsilon,
    ).to(device)

def inject_mhc_lite(model, method_cfg): 
    device = next(model.parameters()).device
    freeze_base_model(model)
    return olmo_mhc_lite(model, 
            num_streams=method_cfg.shc_num_streams,
            num_fracs=method_cfg.mhc_num_fracs,
            ablate_mapping = method_cfg.shc_ablation_mapping
        ).to(device)
    
def inject_kromHC(model, method_cfg): 
    device = next(model.parameters()).device
    freeze_base_model(model)
    return olmo_kromhc(model, 
            num_streams = method_cfg.shc_num_streams,
            num_fracs = method_cfg.mhc_num_fracs,
            ablate_mapping = method_cfg.shc_ablation_mapping,
            hres_init = method_cfg.mhc_init_hres,
            hres_init_noise_std = method_cfg.mhc_hres_init_noise_std,
        ).to(device)


def inject_static_kromHC(model, method_cfg):
    device = next(model.parameters()).device
    freeze_base_model(model)

    return olmo_static_kromhc(
        model,
        num_streams = method_cfg.shc_num_streams,
        num_fracs = method_cfg.mhc_num_fracs,
        ablate_mapping = method_cfg.shc_ablation_mapping,
        hres_init = method_cfg.mhc_init_hres,
        hres_init_noise_std = method_cfg.mhc_hres_init_noise_std,
    ).to(device)
    
def inject_static_mhc_lite(model, method_cfg):
    device = next(model.parameters()).device
    freeze_base_model(model)

    return olmo_static_mhc_lite(
        model,
        num_streams = method_cfg.shc_num_streams,
        num_fracs = method_cfg.mhc_num_fracs,
        ablate_mapping = method_cfg.shc_ablation_mapping,
        hres_init = method_cfg.mhc_init_hres,
        hres_init_noise_std = method_cfg.mhc_hres_init_noise_std,
    ).to(device)


def shc_lora(model, method_cfg): 
    """ inject SHC then apply LoRA, enabling non-ablated SHC logits """
    model = inject_hyperconnections(model, method_cfg)
    model = apply_lora(model, method_cfg)
    ablated = list(method_cfg.shc_ablation_mapping) 
    for name, param in model.named_parameters():
        if "pre_logits" in name and "pre" not in ablated:
            param.requires_grad = True
        elif "post_logits" in name and "post" not in ablated:
            param.requires_grad = True
        elif "res_logits" in name and "res" not in ablated:
            param.requires_grad = True
    return model 

def shc_vera(model, method_cfg):
    """ inject SHC then apply VeRA, enabling non-ablated SHC logits """
    model = inject_hyperconnections(model, method_cfg)
    model = apply_vera(model, method_cfg)
    ablated = list(method_cfg.shc_ablation_mapping)
    for name, param in model.named_parameters():
        if "pre_logits" in name and "pre" not in ablated:
            param.requires_grad = True
        elif "post_logits" in name and "post" not in ablated:
            param.requires_grad = True
        elif "res_logits" in name and "res" not in ablated:
            param.requires_grad = True
    return model


def kromhc_lora(model, method_cfg):
    """ inject KromHC first, then LoRA inside the wrapped branches

    This baseline routes the full LoRA-modified branch output through KromHC:
    branch(x) = frozen_branch(x) + LoRA_delta(x). (This is *different* from
    just routing the delta!)
    """
    model = inject_kromHC(model, method_cfg)
    model = apply_lora(model, method_cfg)
    enable_kromhc_routing_parameters(model)
    return model


def kromhc_vera(model, method_cfg):
    """ inject KromHC first, then VeRA inside the wrapped branches """
    model = inject_kromHC(model, method_cfg)
    model = apply_vera(model, method_cfg)
    enable_kromhc_routing_parameters(model)
    return model


def static_kromhc_lora(model, method_cfg):
    """ inject static KromHC first, then LoRA inside the wrapped branches """
    model = inject_static_kromHC(model, method_cfg)
    model = apply_lora(model, method_cfg)
    enable_kromhc_routing_parameters(model)
    return model


def get_olmo_decoder_layers(model):
    """ returns the OLMo decoder layers """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("could not find OLMo decoder layers at model.model.layers")


def get_olmo_lm_head(model):
    """ returns the OLMo LM head if available """
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise ValueError("could not find OLMo LM head at model.lm_head")


def get_olmo_final_norm(model):
    """ returns the OLMo final norm if available """
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise ValueError("could not find OLMo final norm at model.model.norm")


def inject_layer_tuning(model, method_cfg):
    """ prepares OLMo modules and applies layer tuning """
    freeze_base_model(model)            # start from fully frozen OLMo
                                        # references real layers of model
    layers = get_olmo_decoder_layers(model)
    num_layers_to_train = method_cfg.layer_tuning_train_last_n_layers
    if num_layers_to_train < 0:
        raise ValueError("layer_tuning_train_last_n_layers must be non-negative")
                                        # slice keeps refs, not layer copies
    selected_layers = []
    if num_layers_to_train > 0:
        selected_layers = layers[-num_layers_to_train:] 

    lm_head = model.lm_head if method_cfg.layer_tuning_train_lm_head else None
    final_norm = model.model.norm if method_cfg.layer_tuning_train_norm else None

    apply_layer_tuning(
        selected_layers,
        lm_head = lm_head,
        final_norm = final_norm,
    )

    return model


def inject_delta_kromhc_adapter(model, method_cfg):
    device = next(model.parameters()).device
    freeze_base_model(model)

    model = olmo_delta_kromhc_adapter(
        model,
        num_streams = method_cfg.shc_num_streams,
        num_fracs = method_cfg.mhc_num_fracs,
        ablate_mapping = method_cfg.shc_ablation_mapping,
        hres_init = method_cfg.mhc_init_hres,
        hres_init_noise_std = method_cfg.mhc_hres_init_noise_std,
        gamma_init = method_cfg.delta_kromhc_gamma_init,
        train_gamma = method_cfg.delta_kromhc_train_route_params,
        static_routing = method_cfg.delta_kromhc_static_routing,
        route_mode = getattr(
            method_cfg,
            "delta_kromhc_route_mode",
            "additive",
        ),
        write_only = getattr(
            method_cfg,
            "delta_kromhc_write_only",
            False,
        ),
        lambda_init = getattr(
            method_cfg,
            "delta_kromhc_lambda_init",
            0.1,
        ),
        rho_init = getattr(
            method_cfg,
            "delta_kromhc_rho_init",
            1.0,
        ),
        post_init_offset = getattr(
            method_cfg,
            "delta_kromhc_post_init_offset",
            0.0,
        ),
    ).to(device)

    return model


def delta_kromhc_lora(model, method_cfg):
    model = inject_delta_kromhc_adapter(model, method_cfg)
    model = apply_lora(model, method_cfg)
    enable_delta_kromhc_parameters(
        model,
        static_routing = method_cfg.delta_kromhc_static_routing,
        route_mode = getattr(method_cfg, "delta_kromhc_route_mode", "additive"),
        train_route = method_cfg.delta_kromhc_train_route_params,
        write_only = getattr(method_cfg, "delta_kromhc_write_only", False),
    )
    return model


def delta_kromhc_vera(model, method_cfg):
    model = inject_delta_kromhc_adapter(model, method_cfg)
    model = apply_vera(model, method_cfg)
    enable_delta_kromhc_parameters(
        model,
        static_routing = method_cfg.delta_kromhc_static_routing,
        route_mode = getattr(method_cfg, "delta_kromhc_route_mode", "additive"),
        train_route = method_cfg.delta_kromhc_train_route_params,
        write_only = getattr(method_cfg, "delta_kromhc_write_only", False),
    )
    return model


def delta_kromhc_dora(model, method_cfg):
    model = inject_delta_kromhc_adapter(model, method_cfg)
    model = apply_dora(model, method_cfg)
    enable_delta_kromhc_parameters(
        model,
        static_routing = method_cfg.delta_kromhc_static_routing,
        route_mode = getattr(method_cfg, "delta_kromhc_route_mode", "additive"),
        train_route = method_cfg.delta_kromhc_train_route_params,
        write_only = getattr(method_cfg, "delta_kromhc_write_only", False),
    )
    return model


def delta_kromhc_ia3(model, method_cfg):
    model = inject_delta_kromhc_adapter(model, method_cfg)
    model = apply_ia3(model, method_cfg)
    enable_delta_kromhc_parameters(
        model,
        static_routing = method_cfg.delta_kromhc_static_routing,
        route_mode = getattr(method_cfg, "delta_kromhc_route_mode", "additive"),
        train_route = method_cfg.delta_kromhc_train_route_params,
        write_only = getattr(method_cfg, "delta_kromhc_write_only", False),
    )
    return model


def inject_method(model, cfg):
    """ applies selected finetuning method (cfg) to given model """
    method_name = cfg.method.selected_method.lower()

    if method_name in ["none", "base", "frozen"]:
        freeze_base_model(model)        # no trainable params in baseline
        return model
    if method_name == "lora":
        return apply_lora(model, cfg.method)
    if method_name == "ia3":
        return apply_ia3(model, cfg.method)
    if method_name == "dora":
        return apply_dora(model, cfg.method)
    if method_name == "vera": 
        return apply_vera(model, cfg.method)
    if method_name == "prompt_tuning":
        return apply_prompt_tuning(model, cfg.method)
    if method_name == "layer_tuning":
        return inject_layer_tuning(model, cfg.method)
    if method_name == "kromhc_lora":
        return kromhc_lora(model, cfg.method)
    if method_name == "kromhc_vera":
        return kromhc_vera(model, cfg.method)
    if method_name == "static_kromhc_lora":
        return static_kromhc_lora(model, cfg.method)
    if method_name == "delta_kromhc_lora":
        return delta_kromhc_lora(model, cfg.method)
    if method_name == "delta_kromhc_vera":
        return delta_kromhc_vera(model, cfg.method)
    if method_name == "delta_kromhc_dora":
        return delta_kromhc_dora(model, cfg.method)
    if method_name == "delta_kromhc_ia3":
        return delta_kromhc_ia3(model, cfg.method)
    if method_name == "mhc_lite": 
        return inject_mhc_lite(model, cfg.method)
    if method_name == "kromhc":
        return inject_kromHC(model, cfg.method)
    if method_name == "static_kromhc":
        return inject_static_kromHC(model, cfg.method)
    if method_name == "shc":
        return inject_hyperconnections(model, cfg.method)
    if method_name == "static_mhc_lite":
        return inject_static_mhc_lite(model, cfg.method)
    if method_name == "shc_lora": 
        return shc_lora(model, cfg.method)
    if method_name == "shc_vera":
        return shc_vera(model, cfg.method)
    if method_name == "mhc":
        return inject_mhc(model, cfg.method)

    raise ValueError(f"unknown method: {cfg.method.selected_method}")


def count_trainable_parameters(model):
    """ counts total and trainable parameters """
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0.0

    return {
        "trainable_parameters" : trainable,
        "total_parameters" : total,
        "trainable_percentage" : percentage,
    }


def print_trainable_parameters(model, max_names = 20):
    """ prints how many parameters will be trained """
    counts = count_trainable_parameters(model)

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print("=" * 80)
    print(f"trainable parameters: {counts['trainable_parameters']:,}")
    print(f"total parameters:     {counts['total_parameters']:,}")
    print(f"trainable percentage: {counts['trainable_percentage']:.4f}%")
    print(f"first {max_names} trainable parameter(s):")
    
    for name, parameter in trainable_parameters[:max_names]:
        print(f"  - {name}:\n{parameter.detach().cpu().float().numpy()}")

    if len(trainable_parameters) > max_names:
        print(f"  ... and {len(trainable_parameters) - max_names} more")

    print("=" * 80)