from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


class BaseDataIngestor(ABC):
    def __init__(self, root_dir: str, config: dict[str, Any]):
        """Initializes the ingestor, parses the config, and builds the file registry."""
        self.root_dir = Path(root_dir)
        self.config = config
        self.file_registry: pl.DataFrame = None

        # Build the registry immediately upon instantiation
        self._build_registry()

    def _build_registry(self):
        """The orchestrator method that discovers files, pairs them, and assigns splits."""
        records = []

        # ---------------------------------------------------------
        # STEP 1 & 2: Discover and Pair Files based on split/modality
        # ---------------------------------------------------------
        split_sep = self.config.get('split_separation')
        mod_sep = self.config.get('modality_separation')

        if split_sep == 'physical':
            # Search inside specific Train/Val/Test/Fold directories
            split_dirs = self.config.get('split_dirs', {})
            for split_key, split_rel_path in split_dirs.items():
                if not split_key.endswith('_dir'):
                    raise ValueError(f"Invalid split key: {split_key}. Must end with '_dir'")

                clean_split_label = split_key.replace('_dir', '')
                search_base = self.root_dir / split_rel_path

                records.extend(self._scan_and_pair(search_base, mod_sep, split_label=clean_split_label))

        else:
            # Search the whole root directory (Regex or None)
            records.extend(self._scan_and_pair(self.root_dir, mod_sep, split_label=None))

        # Convert to Polars DataFrame for fast, vectorized operations later
        df = pl.DataFrame(
            records, schema={'roi_id': pl.Utf8, 'image_path': pl.Utf8, 'mask_path': pl.Utf8, 'split': pl.Utf8}
        )

        # ---------------------------------------------------------
        # STEP 3: Apply Regex Split Tagging (if applicable)
        # ---------------------------------------------------------
        if split_sep == 'filename_regex':
            regex_pattern = self.config.get('split_args', {}).get('regex', '')
            if not regex_pattern:
                raise ValueError('split_args.regex must be provided for filename_regex split_separation.')

            # Polars native regex extraction
            df = df.with_columns(pl.col('image_path').str.extract(regex_pattern, 1).alias('split'))
        elif split_sep == 'none':
            df = df.with_columns(pl.lit('unassigned').alias('split'))

        # Drop any rows where split couldn't be determined or file pairing failed
        self.file_registry = df.drop_nulls()

    def _scan_and_pair(self, base_path: Path, mod_sep: str, split_label: str = None) -> list:
        """Handles the actual file discovery and RGB-to-Mask pairing logic."""
        records = []

        if mod_sep == 'bundled_archive':
            # Method 1 (Parquet) - Image and Mask are the exact same file
            for file_path in base_path.rglob('*.parquet'):
                records.append(
                    {
                        'roi_id': file_path.stem,
                        'image_path': str(file_path),
                        'mask_path': str(file_path),  # Points to same archive
                        'split': split_label,
                    }
                )

        elif mod_sep == 'physical_parallel':
            # Method 2, 3, 4, 5 - Images and Masks are in separate places
            mod_dirs = self.config.get('modality_dirs', {})
            img_dir = base_path / mod_dirs.get('image_dir', '')
            mask_dir = base_path / mod_dirs.get('mask_dir', '')

            pairing_rule = self.config.get('modality_pairing_rule', {})
            target_ext = pairing_rule.get('match_extension', '')
            suffix_to_replace = pairing_rule.get('suffix_to_replace', '')
            add_suffix = pairing_rule.get('add_suffix', '')

            # Assume images are PNG/TIF unless specified
            for img_path in img_dir.rglob('*.*'):
                if img_path.suffix not in ['.png', '.tif', '.tiff', '.jpg']:
                    continue

                # Construct expected mask name
                roi_id = img_path.stem
                mask_stem = roi_id

                if suffix_to_replace:
                    mask_stem = mask_stem.replace(suffix_to_replace, '')
                if add_suffix:
                    mask_stem += add_suffix

                expected_mask_name = f'{mask_stem}{target_ext}'
                expected_mask_path = mask_dir / expected_mask_name

                if expected_mask_path.exists():
                    records.append(
                        {
                            'roi_id': roi_id,
                            'image_path': str(img_path),
                            'mask_path': str(expected_mask_path),
                            'split': split_label,
                        }
                    )
                else:
                    print(f'Warning: Missing mask for {img_path.name}. Skipping.')

        return records

    def get_registry(self, split: str = None) -> pl.DataFrame:
        """Returns the registry, optionally filtered by split (train, val, test)."""
        if split:
            return self.file_registry.filter(pl.col('split') == split)
        return self.file_registry

    @abstractmethod
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, dict]:
        """MUST be implemented by subclasses (e.g., ParquetIngestor, GeoJsonIngestor).
        Input: A dictionary representing one row from the file_registry.
        Output: (roi_id, image_array, instance_matrix, category_dict).
        """
        pass
