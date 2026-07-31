from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Type

from dacite import from_dict
from torch import nn


@dataclass
class GroupParams:
    hidden_sizes: List[int]

    @abstractmethod
    def get_module_class(self) -> Type[nn.Module]:
        pass

@dataclass
class SensoryParams(GroupParams):
    recurrence_active: Optional[bool]

    def get_module_class(self) -> Type[nn.Module]:
        from models.CfcModel import CfcSensory
        return CfcSensory

@dataclass
class InterParams(GroupParams):
    feedback_loop: Optional[bool]
    sparsity: Optional[float]
    excitatory_ratio: Optional[float]
    use_polarity: Optional[bool]

    def get_module_class(self) -> Type[nn.Module]:
        from models.CfcModel import CfcInter
        return CfcInter

@dataclass
class CommandParams(GroupParams):
    sparsity: Optional[float]
    excitatory_ratio: Optional[float]
    use_polarity: Optional[bool]

    def get_module_class(self) -> Type[nn.Module]:
        from models.CfcModel import CfcInter
        return CfcInter

@dataclass
class MotorParams(GroupParams):
    recurrence_active: Optional[bool]
    use_polarity: Optional[bool]

    def get_module_class(self) -> Type[nn.Module]:
        from models.CfcModel import CfcMotor
        return CfcMotor

@dataclass
class LayerParams:
    layer: GroupParams

@dataclass
class ModelParams:
    embedding_dim: int
    layers: Dict[str, LayerParams] #  # str здесь это по сути индекс/название слоя – в словарях это все равно всегда строка
    seed: int


TYPE_MAP = {
    "sensory": SensoryParams,
    "inter": InterParams,
    "command": CommandParams,
    "motor": MotorParams
}


def parse_model_layers(raw_dict: dict) -> ModelParams:
    processed_layers = {}
    for name, layer_data in raw_dict["layers"].items():
        layer_inner = layer_data["layer"]
        layer_type = layer_inner.pop("type")

        target_class = TYPE_MAP[layer_type]

        clean_inner = {k: v for k, v in layer_inner.items() if v is not None}
        group_obj = from_dict(data_class=target_class, data=clean_inner)

        processed_layers[name] = {
            "layer": group_obj,
        }

    final_dict = {
        "embedding_dim": raw_dict["embedding_dim"],
        "seed": raw_dict["seed"],
        "layers": processed_layers
    }

    return from_dict(data_class=ModelParams, data=final_dict)