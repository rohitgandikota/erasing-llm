# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from safetensors.torch import save_file


LORA_PREFIX = "lora"



TRAINING_METHODS = Literal[
    "attn",  # train all attn layers
    "mlp",  # train all mlp layers
    "full",  # train all layers
]


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=1,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    def __init__(
        self,
        model,
        layer_ids,
        rank: int = 1,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        train_method: TRAINING_METHODS = "full",
        layer_filter = None,
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha


        self.module = LoRAModule


        self.model_loras = self.create_modules(
            LORA_PREFIX,
            model,
            layer_ids,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
            layer_filter=layer_filter,
        )
        print(f"create LoRA for model: {len(self.model_loras)} modules.")


        lora_names = set()
        for lora in self.model_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)


        for lora in self.model_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del model

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix,
        model,
        layer_ids,
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
        layer_filter,
    ) -> list:
        loras = []
        names = []
        for layer_id in layer_ids:
            for name, module in (model.model.layers[layer_id].named_modules()):
                if layer_filter is not None:
                    if layer_filter not in name:
                        continue
                if 'attn' in train_method:
                    if 'attn' not in name:
                        continue
                elif 'mlp' in train_method:
                    if 'mlp' not in name:
                        continue
                elif train_method == 'full':
                    pass
                else:
                    raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
                    
                if module.__class__.__name__ == 'Linear':
                    lora_name = prefix + "." + str(layer_id) + "." + name
                    lora_name = lora_name.replace(".", "-")
                    lora = self.module(
                        lora_name, module, multiplier, rank, self.alpha
                    )
                    if lora_name not in names:
                        loras.append(lora)
                        names.append(lora_name)
                        # print(lora_name)
        return loras

    def prepare_optimizer_params(self):
        all_params = []

        if self.model_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.model_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

#         for key in list(state_dict.keys()):
#             if not key.startswith("lora"):
#                 # lora以外除外
#                 del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    def set_scale(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.model_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.model_loras:
            lora.multiplier = 0