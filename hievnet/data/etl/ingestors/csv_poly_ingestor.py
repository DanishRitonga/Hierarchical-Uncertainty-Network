import cv2
import numpy as np
import polars as pl

from ._base import BaseDataIngestor


class CSVPolygonIngestor(BaseDataIngestor):
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Fetch column mapping
        col_map = self.config.get('csv_column_map', {})
        col_x = col_map.get('x_coords')
        col_y = col_map.get('y_coords')
        col_cat = col_map.get('category')

        if not all([col_x, col_y, col_cat]):
            raise KeyError('Missing required csv_column_map keys. Need x_coords, y_coords, and category.')

        # 2. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        h, w = image_array.shape[:2]

        # 3. Read the CSV using Polars
        try:
            df = pl.read_csv(mask_path)
        except Exception as e:
            raise ValueError(f'Failed to read CSV at {mask_path}: {e}')

        # 4. Extract Bounding Boxes from Polygon Coordinates
        bboxes = []

        for cell_row in df.iter_rows(named=True):
            # Grab the comma-separated strings
            x_str = cell_row[col_x]
            y_str = cell_row[col_y]

            # Skip empty or malformed rows
            if not x_str or not y_str:
                continue

            # Split the strings and cast directly to an integer numpy array
            x_arr = np.array(x_str.split(','), dtype=np.int32)
            y_arr = np.array(y_str.split(','), dtype=np.int32)

            # Zip them together into the (N, 2) shape OpenCV expects
            pts = np.column_stack((x_arr, y_arr))

            # Extract bounding box directly from polygon coordinates
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(pts)

            # Extract and Standardize the Category
            raw_category = cell_row[col_cat]
            standardized_category = self.standardize_label(raw_category)

            bboxes.append([x_bbox, y_bbox, x_bbox + w_bbox, y_bbox + h_bbox, standardized_category])

        # 5. Safe bounding box array initialization with boundary clipping and degenerate box filtering
        if len(bboxes) > 0:
            bboxes_array = np.array(bboxes, dtype=np.int32)

            # Clip X coordinates (xmin at index 0, xmax at index 2) to [0, w]
            bboxes_array[:, [0, 2]] = np.clip(bboxes_array[:, [0, 2]], 0, w)

            # Clip Y coordinates (ymin at index 1, ymax at index 3) to [0, h]
            bboxes_array[:, [1, 3]] = np.clip(bboxes_array[:, [1, 3]], 0, h)

            # Filter out degenerate boxes (where area became 0 after clipping)
            valid_boxes = (bboxes_array[:, 2] > bboxes_array[:, 0]) & (bboxes_array[:, 3] > bboxes_array[:, 1])
            bboxes_array = bboxes_array[valid_boxes]

        else:
            bboxes_array = np.empty((0, 5), dtype=np.int32)

        tissue_origin = self.resolve_tissue()

        return (roi_id, image_array, bboxes_array, tissue_origin)
