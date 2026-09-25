
import fiftyone as fo
import fiftyone.core.labels as fol
from PIL import Image
from tqdm import tqdm
import os
import yaml


def load_dl_dataset(
        root_dir, 
        name, 
        type,
        overwrite=True):

    if overwrite and name in fo.list_datasets():
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)

    # Load YOLO-Pose class names if keypoints exist
    if type == "kpt":
        yaml_path = os.path.join(root_dir, "dataset.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        class_names = cfg["names"]
    else:
        class_names = {}

    for split in ["train", "val"]:

        if type == "seg": 
            data_dir = os.path.join(root_dir, split, "data")
            label_dir = os.path.join(root_dir, split, "labels")

        elif type == "kpt":
            data_dir = os.path.join(root_dir, "images", split)
            label_dir = os.path.join(root_dir, "labels", split)

        file_list = [f for f in os.listdir(data_dir) if f.endswith(".png")]

        samples = []

        for fname in tqdm(file_list, desc=f"Loading {split}"):
            image_path = os.path.join(data_dir, fname)

            sample = fo.Sample(filepath=image_path)
            sample["split"] = split

            # segmentation 
            mask_path = os.path.join(label_dir, fname)
            if os.path.exists(mask_path):
                sample["ground_truth"] = fol.Segmentation(mask_path=mask_path)

            # keypoints
            kpt_file = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(kpt_file):

                yaml_path = os.path.join(root_dir, "dataset.yaml")
                with open(yaml_path) as f:
                    cfg = yaml.safe_load(f)
                class_names = cfg["names"]

                keypoints_list = []
                with open(kpt_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        cls_id = int(parts[0])
                        label_name = class_names[cls_id]

                        x_norm = float(parts[1])
                        y_norm = float(parts[2])

                        keypoints_list.append(
                            fol.Keypoint(
                                label=label_name,
                                points=[(x_norm, y_norm)]
                            )
                        )

                if keypoints_list:
                    sample["keypoints"] = fol.Keypoints(keypoints=keypoints_list)

            samples.append(sample)

        dataset.add_samples(samples)

    return dataset

def load_segmentation_predictions(
        image_dir,
        gt_dir,
        pred_dir,
        name,
        overwrite=True):
    """
    Load images together with ground-truth and prediction segmentation
    masks into a FiftyOne dataset.

    Assumes that image, GT mask, and prediction mask filenames correspond
    exactly.

    Example:
        image_dir/image001.png
        gt_dir/image001.png
        pred_dir/image001.png
    """

    if overwrite and name in fo.list_datasets():
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)

    image_dir = os.path.abspath(image_dir)
    gt_dir = os.path.abspath(gt_dir)
    pred_dir = os.path.abspath(pred_dir)

    file_list = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith(".png")
    )

    samples = []

    for fname in tqdm(file_list, desc="Loading evaluation dataset"):

        image_path = os.path.join(image_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        # Require both masks
        if not os.path.exists(gt_path):
            print(f"WARNING: missing annotation: {fname}")
            continue

        if not os.path.exists(pred_path):
            print(f"WARNING: missing prediction: {fname}")
            continue

        sample = fo.Sample(filepath=image_path)

        sample["ground_truth"] = fol.Segmentation(
            mask_path=gt_path
        )

        sample["prediction"] = fol.Segmentation(
            mask_path=pred_path
        )

        samples.append(sample)

    dataset.add_samples(samples)

    return dataset
