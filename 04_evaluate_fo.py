from pathlib import Path
import numpy as np
from PIL import Image

import fiftyone as fo
import fiftyone.core.labels as fol


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

IMAGE_DIR = Path(
    r"O:\Data-Work\22_Plant_Production-CH\224_Digitalisation\Jonas_Anderegg_Files"
    r"\B_Data\01_DL_Datasets\03_CANOPY_PROXIMAL_SYMPTOMS\iter2"
    r"\dataset_src\segmentations_export\data"
)

GT_DIR = Path(
    r"O:\Data-Work\22_Plant_Production-CH\224_Digitalisation\Jonas_Anderegg_Files"
    r"\B_Data\01_DL_Datasets\03_CANOPY_PROXIMAL_SYMPTOMS\iter2"
    r"\dataset_src\segmentations_export\labels"
)

PRED_DIR = Path(
    r"O:\Data-Work\22_Plant_Production-CH\224_Digitalisation\Jonas_Anderegg_Files"
    r"\E_Work\91_DL\03_CANOPY_PROXIMAL_SYMPTOMS\predictions"
)


# ---------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------

CLASSES = {
    0: "background",
    2: "Leaf_Damage",
    3: "insect_damage",
    4: "Powdery_Mildew",
}

# Class 1 is deliberately absent.


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def calculate_iou_f1(gt, pred, class_id):
    """
    Calculate IoU and F1 for one class for one image.
    """

    gt_class = gt == class_id
    pred_class = pred == class_id

    tp = np.logical_and(gt_class, pred_class).sum()
    fp = np.logical_and(~gt_class, pred_class).sum()
    fn = np.logical_and(gt_class, ~pred_class).sum()

    denominator_iou = tp + fp + fn
    denominator_f1 = 2 * tp + fp + fn

    if denominator_iou == 0:
        iou = np.nan
    else:
        iou = tp / denominator_iou

    if denominator_f1 == 0:
        f1 = np.nan
    else:
        f1 = 2 * tp / denominator_f1

    return float(iou), float(f1)


def load_mask(path):
    """
    Load a class-ID mask.
    """
    return np.array(Image.open(path))


# ---------------------------------------------------------------------
# Create FiftyOne dataset
# ---------------------------------------------------------------------

dataset_name = "canopy_proximal_symptoms_eval"

# Delete an existing dataset with this name so that the script can
# simply be rerun.
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)


# ---------------------------------------------------------------------
# Match images, GT masks and predictions
# ---------------------------------------------------------------------

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
)

print(f"Found {len(image_paths)} images")


for image_path in image_paths:

    gt_path = GT_DIR / image_path.name
    pred_path = PRED_DIR / image_path.name

    if not gt_path.exists():
        print(f"WARNING: missing annotation: {image_path.name}")
        continue

    if not pred_path.exists():
        print(f"WARNING: missing prediction: {image_path.name}")
        continue

    # Load masks
    gt = load_mask(gt_path)
    pred = load_mask(pred_path)

    # Basic validation
    gt_values = np.unique(gt)
    pred_values = np.unique(pred)

    unexpected_gt = set(gt_values) - set(CLASSES)
    unexpected_pred = set(pred_values) - set(CLASSES)

    if unexpected_gt:
        print(
            f"WARNING {image_path.name}: "
            f"unexpected GT values {unexpected_gt}"
        )

    if unexpected_pred:
        print(
            f"WARNING {image_path.name}: "
            f"unexpected prediction values {unexpected_pred}"
        )

    if gt.shape != pred.shape:
        print(
            f"WARNING {image_path.name}: "
            f"GT shape {gt.shape} != prediction shape {pred.shape}"
        )
        continue

    # -----------------------------------------------------------------
    # Per-class metrics
    # -----------------------------------------------------------------

    metrics = {}

    for class_id, class_name in CLASSES.items():

        iou, f1 = calculate_iou_f1(
            gt,
            pred,
            class_id,
        )

        metrics[f"iou_{class_name}"] = iou
        metrics[f"f1_{class_name}"] = f1

    # -----------------------------------------------------------------
    # Overall metrics
    #
    # Here we calculate macro averages across the classes.
    # Background is included for now; we can change this easily.
    # -----------------------------------------------------------------

    ious = [
        metrics[f"iou_{name}"]
        for name in CLASSES.values()
        if not np.isnan(metrics[f"iou_{name}"])
    ]

    f1s = [
        metrics[f"f1_{name}"]
        for name in CLASSES.values()
        if not np.isnan(metrics[f"f1_{name}"])
    ]

    overall_iou = np.mean(ious)
    overall_f1 = np.mean(f1s)

    # -----------------------------------------------------------------
    # FiftyOne sample
    # -----------------------------------------------------------------

    sample = fo.Sample(
        filepath=str(image_path)
    )

    sample["ground_truth"] = fol.Segmentation(
        mask_path=str(gt_path)
    )

    sample["prediction"] = fol.Segmentation(
        mask_path=str(pred_path)
    )

    # Overall metrics
    sample["overall_iou"] = float(overall_iou)
    sample["overall_f1"] = float(overall_f1)

    # Class-specific metrics
    for key, value in metrics.items():
        if np.isnan(value):
            sample[key] = None
        else:
            sample[key] = value

    dataset.add_sample(sample)


# ---------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------

print()
print(f"Added {len(dataset)} samples")
print()

dataset.compute_metadata()

print("Dataset created:")
print(dataset)


# ---------------------------------------------------------------------
# Launch FiftyOne
# ---------------------------------------------------------------------

session = fo.launch_app(dataset)

print()
print("FiftyOne App launched.")
print("Press Ctrl+C to stop the Python process.")

session.wait()