# %%
from pathlib import Path

import polars as pl

from hievnet.data.ETL.ingestor import BaseDataIngestor


# %% 1. Create a dummy subclass for testing
class DummyCsvIngestor(BaseDataIngestor):
    def process_item(self, row: dict):
        # We don't care about extracting pixels yet, just testing the registry
        pass


# 2. Define the PanopTILs config dictionary (normally parsed from YAML)
panoptils_config = {
    'ingestion_method': 5,
    'split_separation': 'physical',
    'split_dirs': {'train_dir': 'BootstrapNucleiManualRegions_TCGA/tcga', 'val_dir': 'ManualNucleiManualRegions'},
    'modality_separation': 'physical_parallel',
    'modality_dirs': {'image_dir': 'rgbs', 'mask_dir': 'csv'},
    'modality_pairing_rule': {'match_extension': '.csv'},
}


if __name__ == '__main__':
    # 3. Instantiate the Dummy Ingestor pointing to your actual folder
    # Make sure this path points to where your PanopTILs folder is located
    dataset_root = Path(__file__).parent.parent.joinpath('dataset', 'PanopTILs')

    print(f'Scanning {dataset_root}...')
    try:
        ingestor = DummyCsvIngestor(root_dir=dataset_root, config=panoptils_config)
    except Exception as e:
        print(f'Failed to initialize ingestor: {e}')

    # 4. Fetch the full registry and print the summary
    df_full = ingestor.get_registry()

    print('\n' + '=' * 50)
    print('✅ REGISTRY SUCCESSFULLY BUILT')
    print('=' * 50)

    # Configure Polars to show full strings in the terminal without truncating
    with pl.Config(fmt_str_lengths=100, tbl_rows=10):
        print('\n--- Full Dataset Preview ---')
        print(df_full)

        # 5. Test the split filtering
        print('\n--- Split Value Counts ---')
        print(df_full.group_by('split').len())

        print("\n--- Querying 'train' split only ---")
        df_train = ingestor.get_registry(split='train')
        print(df_train.head(3))

        print("\n--- Querying 'val' split only ---")
        df_val = ingestor.get_registry(split='val')
        print(df_val.head(3))
