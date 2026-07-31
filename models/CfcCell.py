from abc import ABC
from typing import Tuple, List, Final

import torch
import torch.nn as nn

class CfcCell(nn.Module):
    combined_size: int
    in_features: int
    hidden_size: int

    f_func: nn.Linear
    g_func: nn.Linear
    h_func: nn.Linear

    sigmoid: nn.Sigmoid

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size

        self.combined_size = in_features + hidden_size

        self.f_func = nn.Linear(self.combined_size, hidden_size)
        self.g_func = nn.Linear(self.combined_size, hidden_size)
        self.h_func = nn.Linear(self.combined_size, hidden_size)

        self.sigmoid = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.f_func.weight)
        torch.nn.init.constant_(self.f_func.bias, 0.5)

        #----------------сомнительный----------------#
        torch.nn.init.xavier_uniform_(self.g_func.weight)
        torch.nn.init.constant_(self.g_func.bias, 0.0)
        torch.nn.init.xavier_uniform_(self.h_func.weight)
        torch.nn.init.constant_(self.h_func.bias, 0.0)

    def forward(self, inp: torch.Tensor, hx: torch.Tensor, ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([inp, hx], dim=1)

        f_val = self.f_func(combined)
        g_val = self.g_func(combined)
        h_val = self.h_func(combined)

        time_gate = self.sigmoid(-f_val*ts)
        new_hidden = time_gate*g_val + (1.0-time_gate)*h_val

        return new_hidden, new_hidden

class CfcGroup(nn.Module, ABC):
    in_features: int
    out_features: int
    hidden_sizes: List[int]

    layers: nn.ModuleList

    device: Final[torch.device]
    generator: torch.Generator

    def __init__(self, in_features: int,
                 hidden_sizes: List[int], device: torch.device):
        super().__init__()

        self.in_features: Final[int] = in_features
        self.hidden_sizes: Final[List[int]] = hidden_sizes
        self.out_features: Final[int] = hidden_sizes[-1]
        self.device= device

        self.layers = self._build_layers()

    def _build_layers(self) -> nn.ModuleList:
        prev_size = self.in_features
        layers = nn.ModuleList()
        for size in self.hidden_sizes:
            layers.append(CfcCell(prev_size, size))
            prev_size = size
        return layers

    def apply_wiring(self):
        return None

    def forward(self, inp: torch.Tensor,
                prev_hx_states: List[torch.Tensor] | Tuple[torch.Tensor, ...],
                ts: torch.Tensor) -> Tuple[torch.Tensor, list]:

        new_hx_states = []

        state = (inp, prev_hx_states[0], ts)
        for i, cell in enumerate(self.layers):
            x_i, hx = cell(*state)
            state = (x_i, prev_hx_states[i], ts)
            new_hx_states.append(hx)
        output = state[0]
        return output, new_hx_states