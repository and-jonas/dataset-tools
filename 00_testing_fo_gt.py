from pathlib import Path
from src.data.loader import load_dl_dataset
import fiftyone as fo
from fiftyone import ViewField as F
from PIL import Image
import numpy as np
from tqdm import tqdm

# SEGMENTATION ==========================================================================================

# path to data set
dataset_dir = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/01_DL_Datasets/20_Datasets_train/ZenklEtAl2026/EFDv2_segmentation")

# load data set
if "symptoms_dataset" in fo.list_datasets():
    fo.delete_dataset("symptoms_dataset")
dataset = load_dl_dataset(
    root_dir=dataset_dir,
    name="symptoms_dataset",
    type="seg"
    )

# add field indicating which classes are present in each sample
for sample in tqdm(dataset):
    mask = np.array(Image.open(sample.ground_truth.mask_path))
    sample["classes_present"] = np.unique(mask).tolist()
    sample.save()

# # create a dataset view: show only samples matching certain criteria
# dataset.compute_metadata()
# pm = dataset.match(F("classes_present").contains(4))

# make temporary fields to show different classes in different colors for inspection
classes = {
    2: "necrosis",
    3: "insect_damage",
    4: "powdery_mildew"
}

for sample in tqdm(dataset):
    mask = np.array(Image.open(sample.ground_truth.mask_path))    
    for class_id, class_name in classes.items():
        class_mask = (mask == class_id).astype(np.uint8)
        if np.any(class_mask):
            sample[class_name] = fo.Segmentation(mask=class_mask)    
    sample.save()

session = fo.launch_app(dataset)

# session.refresh()
# session.view = pm

# KEYPOINTS ==========================================================================================

dataset_dir = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/01_DL_Datasets/20_Datasets_train/ZenklEtAl2026/EFDv2_segmentation")

# load data set
if "symptoms_dataset" in fo.list_datasets():
    fo.delete_dataset("symptoms_dataset")
dataset = load_dl_dataset(
    root_dir=dataset_dir,
    name="symptoms_dataset",
    type="seg"
    )

one_sample = dataset.first()

session = fo.launch_app(dataset)
