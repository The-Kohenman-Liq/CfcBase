from typing import Dict, Any, List, Callable
from pathlib import Path

class CurriculumManager:
    def __init__(self, eras_config: Dict[str, Any]):
        self.eras = sorted(
            eras_config.items(),
            key=lambda x: x[1].get('max_steps', 0)
        )
        self.num_eras = len(self.eras)
        self.current_era_idx = -1

        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []

    def register_listener(self, callback: Callable[[str, Dict[str, Any]], None]):
        self._listeners.append(callback)

    def step(self, current_epoch: int):
        next_era_idx = self.current_era_idx + 1

        if next_era_idx < self.num_eras:
            era_name, era_cfg = self.eras[next_era_idx]
            max_steps = era_cfg.get('max_steps', float('inf'))

            if current_epoch >= max_steps:
                self.switch_to_era(next_era_idx, era_name, era_cfg)

    def switch_to_era(self, idx: int, name: str, cfg: Dict[str, Any]):
        print(f"\033[33m[Curriculum] Switching to Era: {name} (max_steps: {cfg.get('max_steps')})\033[0m")
        self.current_era_idx = idx

        for listener in self._listeners:
            try:
                listener(name, cfg)
            except Exception as e:
                print(f"\033[33mError in curriculum listener: {e}\033[0m")

    @property
    def current_era_name(self) -> str:
        if self.current_era_idx == -1:
            return "none"
        return self.eras[self.current_era_idx][0]

    @property
    def current_era_config(self) -> Dict[str, Any]:
        if self.current_era_idx == -1:
            return {}
        return self.eras[self.current_era_idx][1]