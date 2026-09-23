import cv2
import fiftyone as fo
import tqdm
import glob
import os
from pathlib import Path
import random
import shutil
import numpy as np
from fiftyone import ViewField as F
from utils.yolov8_export import YOLOv8KeypointExporter


# Segmentations 
def cvat2masks(
        data_src: str, export_dst: str, 
        id_mapping: dict = {1: 'Leaf', 2: 'Leaf_Damage', 3: 'insect_damage', 4:'Powdery_Mildew'}
        ):

    # The order of entries gives the exported priority. First entries have the lowest priority and will be 
    # overwritten by entries which come later
    # classes which are not in the id_mapping will be filtered out and not exported 

    labels_path = str(Path(data_src) / 'annotations.xml')
    data_path = str(Path(data_src) / 'images')

    export_path = Path(export_dst)
    export_path.mkdir(parents=True, exist_ok=True)

    existing_datasets = fo.list_datasets()
    name = "temp_dataset_export"
    if name in existing_datasets:
        dataset = fo.load_dataset(name)
        dataset.delete()

    ds = fo.Dataset.from_dir(
        labels_path=labels_path,
        data_path=data_path,
        dataset_type=fo.types.CVATImageDataset,
        name=name,
    )
    ds.persistent = False

    # Remove unused polylines labels
    label_order = [label_cls for label_cls in id_mapping.values()]
    purged_view = ds.filter_labels(
        "polylines", F("label").is_in(label_order)
    )

    filtered_dataset = fo.Dataset()
    filtered_dataset.add_samples(purged_view)

    # Sort Labels to keep the correct export priority
    for sample in filtered_dataset:
        polylines = sample.polylines.polylines
        sorted_polylines = sorted(polylines, key=lambda pln: label_order.index(pln.label))

        sample.polylines.polylines = sorted_polylines
        sample.save()

    filtered_dataset.export(
        export_dir=export_dst,
        dataset_type=fo.types.ImageSegmentationDirectory,
        label_field='polylines',
        mask_targets=id_mapping
    )

# Keypoints 
def cvat2yolopose(
        data_src: str , 
        export_dst: str, 
        id_mapping: dict = {0: 'Pycnidium', 1: 'Rust'}, class_sizes={0:8, 1:32}, image_size=(1024, 1024)
        ):
    
    labels_path = str(Path(data_src) / 'annotations.xml')
    data_path = str(Path(data_src) / 'images')

    export_path = Path(export_dst)
    export_path.mkdir(parents=True, exist_ok=True)

    existing_datasets = fo.list_datasets()
    name = "temp_dataset_export"
    if name in existing_datasets:
        dataset = fo.load_dataset(name)
        dataset.delete()

    # TODO add filtering of labels

    ds = fo.Dataset.from_dir(
        labels_path=labels_path,
        data_path=data_path,
        dataset_type=fo.types.CVATImageDataset,
        name=name,
    )
    ds.persistent = False

    classes = [cl for cl in id_mapping.values()]

    exporter = YOLOv8KeypointExporter(export_dir=export_path, split='train', classes=classes, class_sizes=class_sizes, image_size=image_size)
    ds.export(dataset_exporter=exporter)

    # TODO automate this later
    print("Do the following manually:")
    print("append 'kpt_shape: [1,3]' to dataset.yaml")
    print("append 'val: ./images/val/' to datase.yaml")

def yolopose2mask(data_src: str , export_dst: str):
    labels_path = Path(data_src) / 'labels'
    data_path = Path(data_src) / 'images'

    export_path = Path(export_dst)
    export_path.mkdir(parents=True, exist_ok=True)

    labels = sorted([lbl for lbl in labels_path.rglob('*') if lbl.is_file()])
    images = sorted([img for img in data_path.rglob('*') if img.is_file()])

    print("converting yolopose .txt labels to masks")

    if len(labels) != len(images):  # some images can be without labels and then no label file is created
        print(f"different amount images {len(images)} than labels {len(labels)} present, searching for the extra images")

        image_filenames = {p.stem for p in images}
        label_filenames = {p.stem for p in labels}
        common_filenames = image_filenames & label_filenames

        # create blank masks for images without labels
        unmatched_image_filenames = image_filenames - label_filenames
        unmatched_images = [p for p in images if p.stem in unmatched_image_filenames]

        for unmatche_image in unmatched_images:
            mask = np.zeros_like(cv2.imread(str(unmatche_image), cv2.IMREAD_GRAYSCALE))
            export_filepath = export_path / (str(unmatche_image.stem) + '.png')
            cv2.imwrite(str(export_filepath), mask)

        images = [p for p in images if p.stem in common_filenames]
        labels = [p for p in labels if p.stem in common_filenames]

    for (label, img) in tqdm.tqdm(zip(labels, images), total=len(labels)):
        # check if the pair is matching
        if label.stem != img.stem:
            raise Exception(f"Image: {str(img)} and Label: {str(label)} are not matching")
        
        mask = np.zeros_like(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))

        # Read the YOLOv8 pose label file
        with open(str(label), "r") as f:
            lines = f.readlines()

        # Process each line in the label file
        for line in lines:
            data = list(map(float, line.strip().split()))
            class_id = int(data[0])  # First value is the class ID
            bbox = data[1:5]         # Bounding box (not used for rendering keypoints here)
            keypoints = data[5:]     # Remaining values are keypoints (x, y, v)

            # Iterate through the keypoints
            for i in range(0, len(keypoints), 3):
                x, y = keypoints[i:i+2]
                # Convert normalized coordinates to image coordinates
                px = int(x * mask.shape[1])
                py = int(y * mask.shape[0])

                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:  # Check bounds
                    mask[py, px] = class_id + 1  # 0 is reserved for background so the class id needs to be offset 

        # save the maks at the appropriate export location
        export_filepath = export_path / (str(label.stem) + '.png')
        cv2.imwrite(str(export_filepath), mask)

# List all relevant CVAT datasets
# Iterate dataset exports to thematical batches based on a desired train / val strategy
BASE_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/01_DL_Datasets/03_CANOPY_PROXIMAL_SYMPTOMS/iter2/dataset_src")
src_directory = BASE_DIR / "src"
directories = [d for d in src_directory.iterdir() if d.is_dir()]

# export normal
for directory in directories:
    cvat2masks(
        data_src=str(directory), 
        export_dst=str(BASE_DIR / "segmentations_export"),
        id_mapping={2: 'Necrosis', 3: 'PhysicalDamage', 4:'PowderyMildew'})
    cvat2yolopose(str(directory), str(BASE_DIR / "keypoints_export"), image_size=(1024, 1024))

# Do a train / val split
