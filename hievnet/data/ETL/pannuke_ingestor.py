# %%
from pathlib import Path

import polars as pl

# %%
script_dir = Path(__file__).parent
panoptils_dir = script_dir.parent.joinpath(
    'dataset',
    'PanopTILs',
)

ann_dir = panoptils_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA',
    'tcga',
    'csv',
)

ann_dir2 = panoptils_dir.joinpath(
    'ManualNucleiManualRegions',
    'csv',
)

metadata_path = panoptils_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA',
    'region_summary.csv',
)

split_dir = panoptils_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA',
    'train_test_splits',
)

df = pl.read_csv(metadata_path)

# %%
filename_df = []
for file_path in ann_dir.rglob('*.csv'):
    filename = file_path.stem
    filename_df.append({'filename': f'{filename}.png'})

filename_df = pl.DataFrame(filename_df)


filename_df2 = []
for file_path in ann_dir2.rglob('*.csv'):
    filename = file_path.stem
    filename_df2.append({'filename': f'{filename}.png'})

filename_df2 = pl.DataFrame(filename_df2)

# %%
dfs = []
for file in split_dir.rglob('*.csv'):
    df = pl.read_csv(file)
    dfs.append(df)

split_df = pl.concat(dfs)
