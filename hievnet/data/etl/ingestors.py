from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl


class BaseDataIngestor(ABC):
    """Abstract base class for data ingestors.

    Handles file discovery, pairing, and split assignment based on a flexible
    configuration.
    """

    def __init__(self, config: dict[str, Any]):
        """Initializes the ingestor, parses the config, and builds the file registry."""
        self.root_dir = Path(config.get('root_dir'))
        self.config = config
        self.file_registry: pl.DataFrame = None
        self.namespace_map = config.get('namespace_map', {})

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

    def standardize_label(self, raw_label: Any) -> str:
        """Translates a raw dataset label into the globally standardized ontology.

        Enforces a 'fail-loud' strategy if an unmapped label is discovered.
        """
        raw_str = str(raw_label)

        # If no map is provided at all, just pass the string through (useful for testing)
        if not self.namespace_map:
            return raw_str

        # The Fail-Loud Gatekeeper
        if raw_str not in self.namespace_map:
            raise ValueError(
                f"Ontology Mapping Error: Found unknown label '{raw_str}' in dataset. "
                f"Please map this label in the 'namespace_map' config block."
            )

        return str(self.namespace_map[raw_str])

    @abstractmethod
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, dict]:
        """Abstract method to process found data."""
        pass


class ParquetIngestor(BaseDataIngestor):  # noqa: D101
    def process_item(self, row: dict) -> Generator[tuple[str, np.ndarray, np.ndarray, dict], None, None]:
        """Takes a registry row representing a Parquet file, extracts the ROIs,
        and yields them in the standardized Method 3 format.
        """  # noqa: D205
        parquet_path = row['image_path']
        base_roi_name = row['roi_id']  # e.g., "train-00000"

        # 1. Peek at the Schema (Lazy Evaluation)
        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()

        rgb_col, mask_col, cat_col = self._identify_columns(schema)

        if not all([rgb_col, mask_col, cat_col]):
            raise ValueError(
                f'Could not map all columns in {parquet_path}. RGB: {rgb_col}, Masks: {mask_col}, Cats: {cat_col}'
            )

        lf = lf.with_row_index('internal_roi_id')

        # Table 1: RGB Images
        df_rgb = lf.select(['internal_roi_id', rgb_col]).collect()

        # Table 2: Masks & Categories (Exploded)
        df_masks = lf.select(['internal_roi_id', mask_col, cat_col]).explode([mask_col, cat_col]).drop_nulls().collect()

        masks_by_roi = {}
        if not df_masks.is_empty():
            for sub_df in df_masks.partition_by('internal_roi_id'):
                roi_key = sub_df['internal_roi_id'][0]
                masks_by_roi[roi_key] = sub_df

        # Now iterate through the RGB images
        for rgb_row in df_rgb.iter_rows(named=True):
            internal_id = rgb_row['internal_roi_id']

            rgb_struct = rgb_row[rgb_col]
            rgb_bytes = rgb_struct['bytes'] if isinstance(rgb_struct, dict) else rgb_struct

            image_array = self._decode_image(rgb_bytes, is_mask=False)
            h, w = image_array.shape[:2]

            instance_matrix = np.zeros((h, w), dtype=np.int32)
            cats = []

            # Fast O(1) dictionary lookup instead of filtering in a loop
            if internal_id in masks_by_roi:
                roi_masks_df = masks_by_roi[internal_id]

                for instance_id, mask_row in enumerate(roi_masks_df.iter_rows(named=True), start=1):
                    mask_struct = mask_row[mask_col]
                    mask_bytes = mask_struct['bytes'] if isinstance(mask_struct, dict) else mask_struct

                    category = mask_row[cat_col]

                    mask_array = self._decode_image(mask_bytes, is_mask=True)

                    if mask_array.ndim > 2:
                        mask_array = mask_array[:, :, 0]

                    instance_matrix[mask_array > 0] = instance_id
                    cats.append(category)

            cats = np.array(cats, dtype=np.int16)

            global_roi_id = f'{base_roi_name}_roi_{internal_id}'
            yield (global_roi_id, image_array, instance_matrix, cats)

    def _identify_columns(self, schema: pl.Schema) -> tuple[str, str, str]:
        """Dynamically identifies columns based on HuggingFace/Parquet Struct schemas."""
        rgb_col, mask_col, cat_col = None, None, None

        for col_name, dtype in schema.items():
            # Match RGB: Struct with 'bytes' (or raw Binary fallback)
            if isinstance(dtype, pl.Struct) or dtype == pl.Binary:
                rgb_col = col_name

            # Match Masks: List of Structs (or List of Binary fallback)
            elif isinstance(dtype, pl.List) and (isinstance(dtype.inner, pl.Struct) or dtype.inner == pl.Binary):
                mask_col = col_name

            # Match Categories: List of Integers
            elif isinstance(dtype, pl.List) and dtype.inner in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8]:
                cat_col = col_name

        return rgb_col, mask_col, cat_col

    def _decode_image(self, byte_string: bytes, is_mask: bool = False) -> np.ndarray:
        """Helper to convert raw bytes back into numpy arrays using OpenCV."""
        # Convert bytes to a 1D uint8 numpy array
        np_arr = np.frombuffer(byte_string, np.uint8)

        # Decode the array. Use IMREAD_UNCHANGED for masks to preserve exact values (e.g., boolean/binary masks),
        # use IMREAD_COLOR for RGB images to ensure 3 channels.
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        decoded_img = cv2.imdecode(np_arr, flags)

        if decoded_img is None:
            raise ValueError('OpenCV failed to decode the byte array.')

        # OpenCV loads images in BGR format by default. Convert to RGB.
        if not is_mask and len(decoded_img.shape) == 3:
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

        return decoded_img
