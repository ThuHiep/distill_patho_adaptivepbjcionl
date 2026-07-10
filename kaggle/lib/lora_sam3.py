from __future__ import annotations
import math
from typing import Iterable, List, Optional, Set, Tuple
import torch
import torch.nn as nn

class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base
        
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + lora_out * self.scaling

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

DEFAULT_LORA_TARGETS: Set[str] = {
    
    
    
    "linear1", "linear2",
}

def inject_lora(
    model: nn.Module,
    target_module_names: Iterable[str] = DEFAULT_LORA_TARGETS,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    path_must_contain: str = "decoder",
    verbose: bool = True,
) -> Tuple[List[str], int]:
    target_set = set(target_module_names)
    replaced: List[str] = []

    for parent_name, parent in list(model.named_modules()):
        
        if path_must_contain and path_must_contain not in parent_name:
            continue
        for attr_name, child in list(parent.named_children()):
            if attr_name in target_set and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                
                base_device = next(child.parameters()).device
                lora_layer.lora_A.data = lora_layer.lora_A.data.to(base_device)
                lora_layer.lora_B.data = lora_layer.lora_B.data.to(base_device)
                setattr(parent, attr_name, lora_layer)
                full_path = f"{parent_name}.{attr_name}" if parent_name else attr_name
                replaced.append(full_path)

    n_lora_params = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)

    if verbose:
        print(f"Injected LoRA into {len(replaced)} modules.")
        print(f"LoRA trainable params: {n_lora_params:,} "
              f"({n_lora_params/1e6:.2f}M)")
        if len(replaced) > 0:
            print("Sample paths:")
            for p in replaced[:5]:
                print(f"  {p}")
            if len(replaced) > 5:
                print(f"  ... ({len(replaced)-5} more)")

    return replaced, n_lora_params

def freeze_non_lora(model: nn.Module) -> Tuple[int, int]:
    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
    return n_train, n_total

def save_lora_state(model: nn.Module, path: str):
    state = {n: p.detach().cpu()
             for n, p in model.named_parameters()
             if ("lora_A" in n or "lora_B" in n)}
    torch.save(state, path)
    return len(state)

def load_lora_state(model: nn.Module, path: str) -> int:
    state = torch.load(path, map_location="cpu")
    own = dict(model.named_parameters())
    n_loaded = 0
    for k, v in state.items():
        if k in own:
            own[k].data = v.to(own[k].device)
            n_loaded += 1
    return n_loaded
