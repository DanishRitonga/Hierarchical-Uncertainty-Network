# %%
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# %%
data_dir = Path("../data/PanNuke/data")
fold = 1
parquet_file = Path(f"{data_dir}/fold{fold}-00000.parquet")

df = pd.read_parquet(parquet_file)


# %%
def _decode_image_bytes(byte_data):
    image = Image.open(io.BytesIO(byte_data))
    return np.array(image)


# %%
def decode_roi_bytes(df, row_index):
    row = df.iloc[row_index]
    byte_data = row["image"]["bytes"]

    return _decode_image_bytes(byte_data)


# %%
row = df.iloc[0]
raw_bytes = row["image"]["bytes"]

# img = _decode_image_bytes(raw_bytes)
img = decode_roi_bytes(df, 1)

plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.axis("off")  # Hide the axis ticks
plt.title(f"PanNuke H&E Image (Tissue: {row['tissue']})")
plt.show()
