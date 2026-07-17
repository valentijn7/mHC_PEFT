# callbacks.py
""" custom Trainer callbacks for logging in training"""

import os
import json

import torch
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter


class WeightGradStatsCallback(TrainerCallback):
    """
    logs per-layer weight and gradient statistics to TensorBoard at every logging step

    tracked per trainable parameter:
        weights/   {name}/mean|std|min|max   
        gradients/ {name}/mean|std|min|max  
    """

    def __init__(self, model: torch.nn.Module):
        self._model      = model
        self._grad_stats = {}           
        self._writer     = None

        # hook fires on every backward pass and stores the gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda g, n = name: self._grad_stats.update(
                        {n: g.detach().float()}
                    )
                )

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._writer = SummaryWriter(log_dir = args.logging_dir)

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer is not None:
            self._writer.close()

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._writer is None:
            return

        step = state.global_step

        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            w = param.detach().float()
            self._writer.add_scalar(f"weights/{name}/mean", w.mean(), step)
            self._writer.add_scalar(f"weights/{name}/std",  w.std(correction = 0), step)
            self._writer.add_scalar(f"weights/{name}/min",  w.min(),  step)
            self._writer.add_scalar(f"weights/{name}/max",  w.max(),  step)

        for name, grad in self._grad_stats.items():
            self._writer.add_scalar(f"gradients/{name}/mean", grad.mean(), step)
            self._writer.add_scalar(f"gradients/{name}/std",  grad.std(correction = 0), step)
            self._writer.add_scalar(f"gradients/{name}/min",  grad.min(),  step)
            self._writer.add_scalar(f"gradients/{name}/max",  grad.max(),  step)

def _collect_adapter_route_gamma(model: torch.nn.Module):
    """collects all adapter_route_gamma parameters for logging"""
    entries = []

    for name, param in model.named_parameters():
        if "adapter_route_gamma" not in name:
            continue

        value = param.detach().float().cpu()
        scalar = value.mean().item()

        if "attn" in name:
            kind = "attn"
        elif "mlp" in name:
            kind = "mlp"
        else:
            kind = "other"

        entries.append(
            {
                "name": name,
                "kind": kind,
                "value": scalar,
                "abs_value": abs(scalar),
            }
        )

    return entries


class GammaStatsCallback(TrainerCallback):
    """logs adapter_route_gamma values for delta-routed adapter methods"""

    def __init__(self, model: torch.nn.Module, save_json: bool = True):
        self._model = model
        self._writer = None
        self._save_json = save_json
        self._json_dir = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self._writer = SummaryWriter(log_dir = args.logging_dir)

        if self._save_json:
            self._json_dir = os.path.join(args.output_dir, "gamma_stats")
            os.makedirs(self._json_dir, exist_ok = True)

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer is not None:
            self._writer.close()

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._writer is None:
            return

        entries = _collect_adapter_route_gamma(self._model)
        if not entries:
            return

        step = state.global_step

        all_values = torch.tensor([entry["value"] for entry in entries])
        all_abs_values = torch.tensor([entry["abs_value"] for entry in entries])

        self._writer.add_scalar("gamma/adapter_route/mean", all_values.mean(), step)
        self._writer.add_scalar("gamma/adapter_route/std", all_values.std(correction = 0), step)
        self._writer.add_scalar("gamma/adapter_route/min", all_values.min(), step)
        self._writer.add_scalar("gamma/adapter_route/max", all_values.max(), step)
        self._writer.add_scalar("gamma/adapter_route/abs_mean", all_abs_values.mean(), step)

        for kind in ("attn", "mlp", "other"):
            kind_entries = [entry for entry in entries if entry["kind"] == kind]
            if not kind_entries:
                continue

            values = torch.tensor([entry["value"] for entry in kind_entries])
            abs_values = torch.tensor([entry["abs_value"] for entry in kind_entries])

            self._writer.add_scalar(f"gamma/adapter_route/{kind}_mean", values.mean(), step)
            self._writer.add_scalar(f"gamma/adapter_route/{kind}_abs_mean", abs_values.mean(), step)

        for entry in entries:
            safe_name = entry["name"].replace(".", "/")
            self._writer.add_scalar(
                f"gamma/adapter_route/by_parameter/{safe_name}",
                entry["value"],
                step,
            )

        if self._save_json and self._json_dir is not None:
            output = {
                "step": int(step),
                "mean": all_values.mean().item(),
                "std": all_values.std(correction = 0).item(),
                "min": all_values.min().item(),
                "max": all_values.max().item(),
                "abs_mean": all_abs_values.mean().item(),
                "entries": entries,
            }

            path = os.path.join(self._json_dir, f"gamma_stats_step_{step}.json")
            with open(path, "w") as file:
                json.dump(output, file, indent = 2)