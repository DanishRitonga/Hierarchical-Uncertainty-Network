# %%
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# %%
parquet_file = Path("../data/PanNuke/data/fold1-00000-of-00001.parquet")

df = pd.read_parquet(parquet_file)


# %%
def decode_image_bytes(byte_data):
    """Converts raw PNG bytes into a NumPy array."""
    image = Image.open(io.BytesIO(byte_data))
    return np.array(image)


# %%
row = df.iloc[0]
raw_bytes = row["instances"][0]["bytes"]
image_file = io.BytesIO(raw_bytes)

img = Image.open(image_file)

plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis("off")  # Hide the axis ticks
plt.title(f"PanNuke H&E Image (Tissue: {row['tissue']})")
plt.show()
