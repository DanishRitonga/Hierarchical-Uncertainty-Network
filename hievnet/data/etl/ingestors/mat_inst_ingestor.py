import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import find_objects

from ._base import BaseDataIngestor


class MatInstanceIngestor(BaseDataIngestor):
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        h, w = image_array.shape[:2]

        # 2. Load the .mat dictionary using SciPy
        try:
            mat_data = loadmat(mask_path)
        except Exception as e:
            raise ValueError(f'Failed to read .mat file at {mask_path}: {e}')

        # 3. Extract the Instance Matrix
        if 'inst_map' not in mat_data:
            raise KeyError(f"'inst_map' key not found in {mask_path}")

        instance_matrix = mat_data['inst_map'].astype(np.int32)

        # 4. Extract and Standardize Categories
        if 'inst_type' not in mat_data:
            raise KeyError(f"'inst_type' key not found in {mask_path}")

        raw_types = mat_data['inst_type'].flatten()

        cats = [0]
        for raw_cat in raw_types:
            cats.append(self.standardize_label(raw_cat))

        # 5. Extract Bounding Boxes using O(1) Slice Lookups
        # find_objects returns a list of slice tuples where index i corresponds to ID i+1
        slices = find_objects(instance_matrix)
        bboxes = []

        for i, slc in enumerate(slices):
            if slc is None:
                continue  # This instance ID does not exist in the matrix

            instance_id = i + 1
            if instance_id >= len(cats):
                continue

            # slc[0] is the Y-axis slice, slc[1] is the X-axis slice
            ymin, ymax = slc[0].start, slc[0].stop
            xmin, xmax = slc[1].start, slc[1].stop

            class_id = cats[instance_id]
            bboxes.append([xmin, ymin, xmax, ymax, class_id])

        # 6. Safe bounding box array initialization
        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)

            # Defensive clipping (though find_objects inherently stays within array bounds)
            bboxes_array[:, [0, 2]] = np.clip(bboxes_array[:, [0, 2]], 0, w)
            bboxes_array[:, [1, 3]] = np.clip(bboxes_array[:, [1, 3]], 0, h)

            valid_boxes = (bboxes_array[:, 2] > bboxes_array[:, 0]) & (bboxes_array[:, 3] > bboxes_array[:, 1])
            bboxes_array = bboxes_array[valid_boxes]
        else:
            bboxes_array = np.empty((0, 5), dtype=np.int32)

        tissue_origin = self.resolve_tissue()

        return (roi_id, image_array, bboxes_array, tissue_origin)
