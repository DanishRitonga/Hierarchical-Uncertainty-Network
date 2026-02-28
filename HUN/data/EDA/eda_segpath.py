# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'dataset' / 'SegPath' / 'CD3CD20_Lymphocyte'
mask_path = data_dir / 'CD3CD20_Lymphocyte_388_140288_041984_mask.png'


# %%
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
binary_mask = (mask * 255).astype(np.uint8)

contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# %%
bounding_boxes = []
for contour in contours:
    # cv2.boundingRect returns the top-left x,y and the width/height
    x, y, w, h = cv2.boundingRect(contour)

    # Optional: Filter out single-pixel noise
    # if w > 2 and h > 2:

    bounding_boxes.append((x, y, w, h))

# %%
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(binary_mask, cmap='gray')

for x, y, w, h in bounding_boxes:
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

ax.set_title('Bounding Boxes')
plt.tight_layout()
plt.show()
# %%
