# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'dataset' / 'PanopTILs'

csv_filename = 'TCGA-S3-AA15-DX1_xmin55486_ymin28926_MPP-0.2500_xmin-0_ymin-1024_xmax-1024_ymax-2048'

csv_path = data_dir / 'BootstrapNucleiManualRegions_TCGA_1' / 'tcga' / 'csv' / f'{csv_filename}.csv'

img_path = data_dir.joinpath(
    'BootstrapNucleiManualRegions_TCGA_1',
    'tcga',
    'masks',
    'TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_xmin-0_ymin-0_xmax-1024_ymax-1024.png',
)
# %%
df = pd.read_csv(csv_path)

# %%
img = cv2.imread(
    str(img_path),
    cv2.IMREAD_GRAYSCALE,
)

img = np.array(img).astype(np.uint8)

plt.imshow(img == 1)
plt.show()
