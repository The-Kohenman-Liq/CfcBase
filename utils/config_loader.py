import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.dataclasses.model_config import ModelParams, parse_model_layers

RAW_FIELDS = {"num_assoc_commands"}

class ConfigDict(dict):
    def __init__(self, data: Dict[str, Any] = None):
        if data is None:
            data = {}
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                if key in RAW_FIELDS:
                    self[key] = value
                else:
                    self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = [ConfigDict(v) if isinstance(v, dict) else v for v in value]

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

class ProjectConfig:
    def __init__(self, config_dir: str = "configs"):
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent

        self.config_dir = project_root / Path(config_dir)
        self.main_config_path = self.config_dir / "config.yaml"

        if not self.main_config_path.exists():
            raise FileNotFoundError(f"Main config not found at {self.main_config_path}")

        with open(self.main_config_path, 'r') as f:
            raw_main = yaml.safe_load(f)

        self._data = ConfigDict()
        for key, value in raw_main.items():
            if key == "default":
                continue
            if isinstance(value, dict):
                self._data[key] = ConfigDict(value)
            elif isinstance(value, list):
                self._data[key] = [ConfigDict(v) if isinstance(v, dict) else v for v in value]
            else:
                self._data[key] = value
        if "default" in raw_main and isinstance(raw_main["default"], list):
            self._resolve_configs(raw_main["default"], self._data)
        self._data.pop("default", None)

        self._cached_model_params: Optional[ModelParams] = None

    @property
    def model_params(self) -> ModelParams:
        if self._cached_model_params is not None:
            return self._cached_model_params
        try:
            model_ref = self.model
            filename = model_ref.file
            category = model_ref.category if hasattr(model_ref, 'category') else None
        except AttributeError:
            raise ValueError("In config.yaml, you must specify a 'model' section with a 'file' key.")

        raw_model_data = self._load_external_file(category or "", filename)
        if not raw_model_data:
            raise FileNotFoundError(f"Could not load model config file: {filename}")

        self._cached_model_params = parse_model_layers(raw_model_data)
        return self._cached_model_params

    def _resolve_configs(self, raw_source: Any, target_dict: ConfigDict):
        if isinstance(raw_source, list):
            for item in raw_source:
                if isinstance(item, dict):
                    # "- curriculum: eras" and etc case
                    for key, filename in item.items():
                        if key == "_self_":
                            continue
                        resolved_data = self._load_external_file(key, filename)

                        print(f"DEBUG: Ключ в config.yaml='{key}', Файл='{filename}'")

                        if resolved_data is not None:
                            target_dict[key] = ConfigDict(resolved_data)
                else:
                    pass
        elif isinstance(raw_source, dict):
            for k, v in raw_source.items():
                if k == "_self_":
                    continue
                if isinstance(v, dict):
                    target_dict[k] = ConfigDict(v)
                else:
                    target_dict[k] = v

    def _load_external_file(self, category: str, name: str) -> Any:
        for ext in [".yaml", ".yml", ".json"]:
            path1 = self.config_dir / f"{name}{ext}"
            if path1.exists():
                return self._read_file(path1)
            path2 = self.config_dir / category / f"{name}{ext}"
            if path2.exists():
                return self._read_file(path2)
        print(f"Warning: Could not find config file for {category}:{name} (tried {path1} and {path2})")
        return None

    @staticmethod
    def _read_file(path: Path) -> Any:
        with open(path, 'r') as f:
            if path.suffix == '.json':
                return json.load(f)
            return yaml.safe_load(f)

    def __getattr__(self, item):
        return self._data[item]

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        return f"ProjectConfig({dict(self._data)})"
