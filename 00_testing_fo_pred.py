from pathlib import Path

from src.data.loader import load_segmentation_predictions

import fiftyone as fo
from fiftyone import ViewField as F

from PIL import Image
import numpy as np
from tqdm import tqdm


# =====================================================================
# PATHS
# =====================================================================

image_dir = Path(
    r"O:/Data-Work/22_Plant_Production-CH/"
    r"224_Digitalisation/Jonas_Anderegg_Files/"
    r"B_Data/01_DL_Datasets/03_CANOPY_PROXIMAL_SYMPTOMS/"
    r"iter2/dataset_src/segmentations_export/data"
)

gt_dir = Path(
    r"O:/Data-Work/22_Plant_Production-CH/"
    r"224_Digitalisation/Jonas_Anderegg_Files/"
    r"B_Data/01_DL_Datasets/03_CANOPY_PROXIMAL_SYMPTOMS/"
    r"iter2/dataset_src/segmentations_export/labels"
)

pred_dir = Path(
    r"O:/Data-Work/22_Plant_Production-CH/"
    r"224_Digitalisation/Jonas_Anderegg_Files/"
    r"E_Work/91_DL/03_CANOPY_PROXIMAL_SYMPTOMS/"
    r"predictions"
)


# =====================================================================
# LOAD DATASET
# =====================================================================

dataset = load_segmentation_predictions(
    image_dir=image_dir,
    gt_dir=gt_dir,
    pred_dir=pred_dir,
    name="symptoms_predictions",
)


print(dataset)
print(f"Number of samples: {len(dataset)}")


# =====================================================================
# CHECK CLASSES PRESENT IN GROUND TRUTH
# =====================================================================

for sample in tqdm(dataset, desc="Reading ground-truth classes"):

    mask = np.array(
        Image.open(sample.ground_truth.mask_path)
    )

    sample["classes_present"] = np.unique(mask).tolist()

    sample.save()


# =====================================================================
# CREATE INDIVIDUAL CLASS MASKS FOR VISUAL INSPECTION
# =====================================================================

classes = {
    2: "leaf_damage",
    3: "insect_damage",
    4: "powdery_mildew",
}


for sample in tqdm(dataset, desc="Creating visualization masks"):

    gt = np.array(
        Image.open(sample.ground_truth.mask_path)
    )

    pred = np.array(
        Image.open(sample.prediction.mask_path)
    )

    for class_id, class_name in classes.items():

        gt_mask = (gt == class_id).astype(np.uint8)
        pred_mask = (pred == class_id).astype(np.uint8)

        # Only add the field if the class occurs in either mask
        if np.any(gt_mask) or np.any(pred_mask):

            sample[f"gt_{class_name}"] = fo.Segmentation(
                mask=gt_mask
            )

            sample[f"pred_{class_name}"] = fo.Segmentation(
                mask=pred_mask
            )

    sample.save()


# =====================================================================
# LAUNCH FIFTYONE
# =====================================================================

session = fo.launch_app(dataset)

print("FiftyOne App launched.")
print(f"Samples: {len(dataset)}")
