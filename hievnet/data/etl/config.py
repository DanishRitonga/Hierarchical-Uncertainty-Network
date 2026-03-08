from pathlib import Path
from typing import Any

import yaml


class ETLConfig:
    """config manager for ETL."""

    def __init__(self, config_path: str, schema_path: str = None):
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path) if schema_path else Path(__file__).parent / 'etl_schema.yaml'

        self.raw_config = self._load_yaml()
        self.schema = self._load_schema()

        self.global_settings = self.raw_config.get('global_settings', {})
        self.datasets = self.raw_config.get('datasets', {})

        self._validate_schema()

    def _load_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f'Configuration file not found: {self.config_path}')
        with open(self.config_path) as file:
            return yaml.safe_load(file)

    def _load_schema(self) -> dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f'Schema file not found: {self.schema_path}')
        with open(self.schema_path) as file:
            return yaml.safe_load(file)

    def _validate_schema(self):
        validation = self.schema.get('validation', {})
        split_config = self.schema.get('split_separation_config', {})
        mod_config = self.schema.get('modality_separation_config', {})

        required_globals = validation.get('required_globals', [])
        dataset_required_keys = validation.get('dataset_required_keys', [])
        valid_split_seps = validation.get('valid_split_seps', [])
        valid_mod_seps = validation.get('valid_mod_seps', [])

        # Validate global settings
        for req in required_globals:
            if req not in self.global_settings:
                raise KeyError(f"Missing required global setting: '{req}'")

        if not self.datasets:
            raise ValueError("No datasets found in configuration under 'datasets:' key.")

        # Validate each dataset
        for dataset_name, d_conf in self.datasets.items():
            # Check required dataset keys
            for req in dataset_required_keys:
                if req not in d_conf:
                    raise KeyError(f"Dataset '{dataset_name}' is missing required key: '{req}'")

            # Validate split_separation
            split_sep = d_conf['split_separation']
            if split_sep not in valid_split_seps:
                raise ValueError(f"Dataset '{dataset_name}': Invalid split_separation")

            split_reqs = split_config.get(split_sep, {})
            for field in split_reqs.get('required_fields', []):
                if field not in d_conf:
                    raise KeyError(f"Dataset '{dataset_name}' missing '{field}'")

            # Validate constraints for split_separation
            constraints = split_reqs.get('constraints', {})
            if 'split_dirs_keys_must_end_with' in constraints and field == 'split_dirs':
                suffix = constraints['split_dirs_keys_must_end_with']
                for key in d_conf.get('split_dirs', {}):
                    if not key.endswith(suffix):
                        raise ValueError(f"Dataset '{dataset_name}': split_dirs key '{key}' must end with '{suffix}'")

            # Validate modality_separation
            mod_sep = d_conf['modality_separation']
            if mod_sep not in valid_mod_seps:
                raise ValueError(f"Dataset '{dataset_name}': Invalid modality_separation")

            mod_reqs = mod_config.get(mod_sep, {})
            for field in mod_reqs.get('required_fields', []):
                if field not in d_conf:
                    raise KeyError(f"Dataset '{dataset_name}' missing '{field}'")

            # Validate constraints for modality_separation
            constraints = mod_reqs.get('constraints', {})
            if 'modality_dirs_must_contain' in constraints and 'modality_dirs' in d_conf:
                required_keys = constraints['modality_dirs_must_contain']
                for key in required_keys:
                    if key not in d_conf['modality_dirs']:
                        raise KeyError(f"Dataset '{dataset_name}' modality_dirs must contain '{key}'")

    def get_dataset_config(self, dataset_name: str) -> dict[str, Any]:
        """Returns the specific configuration block, with the root_dir fully resolved."""
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")

        # --- TWEAK 2: Resolve the paths ---
        d_conf = self.datasets[dataset_name].copy()
        global_root = Path(self.global_settings['root_dir']).resolve()
        dataset_root = Path(d_conf['root_dir'])

        # pathlib magic: if dataset_root is absolute, it ignores global_root.
        resolved_root = global_root.joinpath(dataset_root)
        print(resolved_root)

        # Inject the fully resolved path back into the dictionary
        d_conf['resolved_root_dir'] = str(resolved_root)

        return d_conf

    def get_global_config(self) -> dict[str, Any]:
        """Returns the global settings."""
        return self.global_settings

    def list_datasets(self) -> list[str]:
        """Returns a list of available dataset."""
        return list(self.datasets.keys())
