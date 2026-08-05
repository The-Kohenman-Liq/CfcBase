import inspect
from dataclasses import asdict
from typing import List, Tuple
from typing import override, Dict

import torch
from torch import nn

from models.CfcCell import CfcCell, CfcGroup
from utils.dataclasses.model_config import LayerParams


class CfcMemoryModel(nn.Module):
    vocab_size: int

    embedding_dim: int

    projection_matrix: nn.Buffer
    groups: nn.ModuleList
    mlp: nn.Sequential

    generator: torch.Generator
    def __init__(self, vocab_size: int,
                 embedding_dim: int,
                 h_layers_param: Dict[str, LayerParams],
                 device: torch.device,
                 seed: int = 2222):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.device = device
        self.generator = torch.Generator().manual_seed(seed)

        # TODO! input_layer
        projection_matrix = torch.randn(self.vocab_size, self.embedding_dim,
                                        generator=self.generator)
        self.register_buffer('projection_matrix', projection_matrix)

        self.groups = nn.ModuleList()
        self._index_map: Dict[str, int] = {}
        self._states: List[List[torch.Tensor]] = []

        current_in_features = self.embedding_dim
        for idx, (name, layer_container) in enumerate(h_layers_param.items()):
            group_params = layer_container.layer

            module_class = group_params.get_module_class()
            params_dict = asdict(group_params)


            filtered_params = {k: v for k, v in params_dict.items() if v is not None}

            sig = inspect.signature(module_class.__init__)
            valid_args = sig.parameters.keys()

            if "in_features" in valid_args:
                filtered_params["in_features"] = current_in_features
            elif "input_size" in valid_args:
                filtered_params["input_size"] = current_in_features

            filtered_params["device"] = self.device
            filtered_params["generator"] = self.generator

            filtered_params = {k: v for k, v in filtered_params.items() if k in valid_args}

            #print(f"argument for layer {idx}: {filtered_params}")
            initialized_group = module_class(**filtered_params)

            self.groups.append(initialized_group)
            self._index_map[name] = idx


            if hasattr(initialized_group, "out_features"):
                current_in_features = initialized_group.out_features
            elif hasattr(group_params, "hidden_sizes"):
                current_in_features = group_params.hidden_sizes[-1]
            else:
                raise AttributeError(f"Не удалось определить выходной размер для слоя {name}")

        hidden_final = current_in_features
        hidden_dim_inter = hidden_final

        self.mlp = nn.Sequential(
            nn.Linear(hidden_final, hidden_dim_inter),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_inter, self.vocab_size)
        )



    def reset_states(self, batch_size: int):
        new_states = []
        for group in self.groups:
            if isinstance(group, CfcGroup):
                group_states = [
                    torch.zeros(batch_size, h_size, device=self.device)
                    for h_size in group.hidden_sizes
                ]
                new_states.append(group_states)
            else:
                new_states.append(None)
        self._states = new_states

    def forward(self, x: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, SeqLen]
            ts: [Batch, SeqLen, 1]
            mask: [Batch, SeqLen] (True/False)
        Returns:
            logits: [Batch, VocabSize] – на общем рефакторинге стоит уже переименовать vocab_size
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x_emb = self.projection_matrix[x.long()]

        self.reset_states(batch_size)

        all_outputs = []
        for t in range(seq_len):
            x_t = x_emb[:, t, :]
            ts_t = ts[:, t, :]

            current_input = x_t

            for i, group in enumerate(self.groups):
                if isinstance(group, CfcGroup):
                    group_output, new_group_states = group(
                        current_input,
                        self._states[i],
                        ts_t
                    )
                    self._states[i] = new_group_states
                    current_input = group_output
                else:
                    current_input = group(current_input)
            all_outputs.append(current_input)

        all_outputs = torch.stack(all_outputs, dim=1) #[Batch, SeqLen, HiddenDim]
        last_valid_indices = (mask.sum(dim=1).long() - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=self.device)
        final_states = all_outputs[batch_idx, last_valid_indices]

        logits = self.mlp(final_states)
        return logits

    def scheduler_step(self):
        """Applies wiring (sparsity/polarity masks) to all applicable layers."""
        for group in self.groups:
            if hasattr(group, "apply_wiring"):
                group.apply_wiring()


class CfcSensory(CfcGroup):
    memory: bool = False

    """"
    Должны быть лишены рекурентности в следствии чего по идее работают чисто как слой фильтрации с
    мгновенной реакцией(в силу этого пробрасывать сигнал с них напрямую в command может иметь смысл).
    По хорошему это должен быть стандартный backbone слой, а то такая механика подразумевает подачу константных
    0 и 1 в качестве hx и ts так что cfc ячейка деградирует до линейного слоя, и я хуй знает как это скажется на градиентах,
    но накладные расходы заметны.
    """
    #ZERO_STATES: Final[Tuple[torch.Tensor, ...]]
    device: torch.device

    def __init__(self, in_features: int,
                 hidden_sizes: List[int], device: torch.device,
                 recurrence_active: bool = False):

        super().__init__(in_features = in_features,
                         hidden_sizes = hidden_sizes, device=device)

        self.memory = recurrence_active

    def _reset_states(self, batch_size):
        return tuple(
            torch.zeros(batch_size, h_size, device=self.device)
            for h_size in self.hidden_sizes
        )

    @override
    def forward(self, inp: torch.Tensor,
                prev_hx_states: List[torch.Tensor] | Tuple[torch.Tensor, ...],
                ts: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        if not self.memory:
            prev_hx_states = self._reset_states(inp.size(0))
        ts = torch.ones(inp.size(0), 1, device=self.device)

        return super().forward(inp, prev_hx_states, ts)


class CfcInter(CfcGroup):
    feedback_loop: bool = True # Рекуррентность между слоями круппы(типа текущий слой так же видит память следующего)
    interneuronal_recurrence: bool = False
    """
    не реализовано.
    That is not like CfcInterneuron recurrence – like an inter cfc–neuron recurrence – типо внутри одного слоя
    буквально нейроны соединяться могут. Как именно – вопрос. TODO!
    """
    sparsity: float = 0.5
    """
    Разряженность внутренних связей. sparsity=0 в купе с use_polarity=false влияет на оптимизацию.
    """
    excitatory_ratio: float = 0.75
    use_polarity: bool = True
    """
    Эти геи придумали жестко фиксировать возбуждающие/тормозящие св-ва нейронов. Типо на эффективности обучения
    сказывается хорошо. Здесь реализовано через применение знаковой маски к абсолютному значению проходящего сигнала.
    Детерменизм обеспечивается за счет generator.
    """
    generator: torch.Generator
    use_masks: bool = True

    hidden_count: int
    def __init__(self,
                 in_features: int,
                 hidden_sizes: List[int],
                 device: torch.device,
                 generator: torch.Generator,
                 feedback_loop: bool = True,
                 sparsity: float = 0.5,
                 excitatory_ratio: float = 0.75,
                 use_polarity: bool = True):
        self.feedback_loop = feedback_loop
        self.hidden_count = len(hidden_sizes)

        self.sparsity = sparsity
        self.excitatory_ratio = excitatory_ratio
        self.use_polarity = use_polarity
        self.generator = generator

        if  not use_polarity and sparsity==0.0:
            self.use_masks = False
        else:
            self.use_masks = True

        super().__init__(in_features, hidden_sizes, device)

    @override
    def _build_layers(self) -> nn.ModuleList:
        prev_size = self.in_features
        feedback_size = 0
        layers = nn.ModuleList()

        for i, size in enumerate(self.hidden_sizes):
            if self.feedback_loop:
                feedback_size = self.hidden_sizes[(i+1)%self.hidden_count]
            cell = CfcCell(prev_size+feedback_size, size) # красота :)))

            combined_size = cell.combined_size

            if self.use_masks:
                weight_shape = (size, combined_size)

                if self.sparsity > 0.0:
                    sparsity_mask = (torch.rand(weight_shape, generator=self.generator) > self.sparsity).float()
                else:
                    sparsity_mask = torch.ones(weight_shape)

                if self.use_polarity:
                    sign_mask = torch.where(
                        torch.rand(weight_shape, generator=self.generator) < self.excitatory_ratio,
                        1.0,
                        -1.0
                    )
                else:
                    sign_mask = torch.ones(weight_shape)

                cell.register_buffer("sparsity_mask", sparsity_mask.to(self.device))
                cell.register_buffer("sign_mask", sign_mask.to(self.device))

            layers.append(cell)
            prev_size=size

        return layers

    @override
    def forward(self, inp: torch.Tensor,
                prev_hx_states: List[torch.Tensor] | Tuple[torch.Tensor, ...],
                ts: torch.Tensor) -> Tuple[torch.Tensor, list]:



        if self.feedback_loop:
            hx_states = tuple(
                torch.cat([hx, prev_hx_states[(i+1)%self.hidden_count]], dim=1)
                for i, hx in enumerate(prev_hx_states)
            )
        else:
            hx_states = prev_hx_states

        return super().forward(inp, hx_states, ts)

    # есть вероятность, что его надо дергать извне после optimiser.step
    @override
    def apply_wiring(self):
        if self.use_masks:
            for cell in self.layers:
                sparsity_mask = cell.sparsity_mask
                sign_mask= cell.sign_mask

                for func in [cell.f_func, cell.g_func, cell.h_func]:
                    with torch.no_grad():
                        if self.use_polarity:
                            func.weight.copy_(func.weight * sign_mask * sparsity_mask)
                        else:
                            func.weight.copy_(func.weight * sparsity_mask)


# А я б честно CfcInter просто настроил, одна херня межнейронной связи нет ни здесь ни там пока что
class CfcCommand(CfcGroup):
    ...

# мне предварительно лень выстраивать из этого отдельный класс, так что пока наследие от CfcSensory.
# TODO!
class CfcMotor(CfcSensory):
    """
    внутренняя рукурентность отсутствует.
    """
    def __init__(self,
                 in_features: int,
                 hidden_sizes: List[int], #should be only one
                 device: torch.device,
                 generator: torch.Generator,
                 recurrence_active: bool = False,
                 use_polarity: bool = True):
        super().__init__(in_features=in_features, hidden_sizes=[hidden_sizes[0]],
                         device=device, recurrence_active=recurrence_active)

        self.use_polarity = use_polarity
        self.generator = generator

        self._build_motor_masks()

    def _build_motor_masks(self):
        if self.use_polarity:
            cell = self.layers[0]
            combined_size = cell.combined_size
            weight_shape = (self.output_size, combined_size)

            sign_mask = torch.where(
                torch.rand(weight_shape, generator=self.generator) < 0.75,
                1.0,
                -1.0
            )
            cell.register_buffer("sign_mask", sign_mask.to(self.device))

    @override
    def forward(self, inp: torch.Tensor,
                prev_hx_states: List[torch.Tensor] | Tuple[torch.Tensor, ...],
                ts: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        if not self.memory:
            prev_hx_states = self._reset_states(inp.size(0))
        #ts = torch.ones(inp.size(0), 1, device=self.device)

        return CfcGroup.forward(self, inp, prev_hx_states, ts) # mda.

    @override
    def apply_wiring(self):
        if self.use_polarity:
            cell = self.layers[0]
            sign_mask = cell.sign_mask
            for func in [cell.f_func, cell.g_func, cell.h_func]:
                with torch.no_grad():
                    func.weight.copy_(func.weight.abs() * sign_mask)
