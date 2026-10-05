
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def split_dataset(
    metadata,
    val_fraction=0.2,
    strategy="random",
    group_column=None,
    val_groups=None,
    seed=42,
):

    df = metadata.copy()
    rng = np.random.default_rng(seed)

    # if strategy != "group":
    #     raise ValueError("Only strategy='group' is supported.")

    if isinstance(group_column, list):
        if len(group_column) > 1:
            raise ValueError("Only one grouping column is supported.")
        group_column = group_column[0]

    # =========================================================
    # RANDOM SPLIT
    # =========================================================

    if strategy == "random":
        n_val = max(1, round(len(df) * val_fraction))
        val_indices = rng.choice(df.index, size=n_val, replace=False)
        df["split"] = "train"
        df.loc[val_indices, "split"] = "val"

    # =========================================================
    # GENOTYPE
    # =========================================================

    elif group_column == "genotype_name":

        # Genotypes actually present in each image
        sample_genotypes = df.apply(
            lambda row: {
                str(g).strip()
                for g in [
                    row["genotype_name1"],
                    row["genotype_name2"],
                ]
                if pd.notna(g) and str(g).strip()
            },
            axis=1,
        )

        # -----------------------------------------------------
        # Build connected genotype groups
        # -----------------------------------------------------

        genotype_groups = []

        for genotypes in sample_genotypes:

            overlapping = [
                group
                for group in genotype_groups
                if group & genotypes
            ]

            if overlapping:

                merged = set(genotypes)

                for group in overlapping:
                    merged |= group
                    genotype_groups.remove(group)

                genotype_groups.append(merged)

            else:
                genotype_groups.append(set(genotypes))

        genotype_groups = [
            group for group in genotype_groups if group
        ]

        # -----------------------------------------------------
        # Determine number of images represented by each
        # genotype group
        # -----------------------------------------------------

        group_sizes = []

        for group in genotype_groups:

            mask = sample_genotypes.apply(
                lambda x: bool(x & group)
            )

            group_sizes.append(mask.sum())

        # -----------------------------------------------------
        # Select groups until approximately val_fraction
        # of images are validation
        # -----------------------------------------------------

        target = len(df) * val_fraction

        group_indices = list(range(len(genotype_groups)))
        rng.shuffle(group_indices)

        selected_groups = []
        n_val = 0

        # Greedily add groups while approaching target
        for i in group_indices:

            size = group_sizes[i]

            if n_val + size <= target:
                selected_groups.append(i)
                n_val += size

        # If nothing was selected, use the smallest group
        if not selected_groups:
            selected_groups = [
                min(
                    group_indices,
                    key=lambda i: group_sizes[i]
                )
            ]
            n_val = group_sizes[selected_groups[0]]

        # -----------------------------------------------------
        # Validation genotypes
        # -----------------------------------------------------

        val_groups = set().union(
            *(genotype_groups[i] for i in selected_groups)
        )

        # -----------------------------------------------------
        # Assign images
        # -----------------------------------------------------

        val_mask = sample_genotypes.apply(
            lambda x: bool(x & val_groups)
        )

        df["split"] = "train"
        df.loc[val_mask, "split"] = "val"

        # -----------------------------------------------------
        # Report
        # -----------------------------------------------------

        val_genotypes = set().union(
            *sample_genotypes[val_mask]
        ) if val_mask.any() else set()

        train_genotypes = set().union(
            *sample_genotypes[~val_mask]
        ) if (~val_mask).any() else set()

        overlap = val_genotypes & train_genotypes

        print(
            f"Validation genotypes ({len(val_genotypes)}):"
        )
        print(sorted(val_genotypes))

        print(
            f"\nTraining genotypes ({len(train_genotypes)}):"
        )
        print(sorted(train_genotypes))

        print(
            f"\nValidation images: "
            f"{val_mask.sum()} / {len(df)} "
            f"({val_mask.mean():.1%})"
        )

        print(
            f"Target validation fraction: "
            f"{val_fraction:.1%}"
        )

        if overlap:
            raise RuntimeError(
                "Genotype leakage detected: "
                f"{sorted(overlap)}"
            )

    # =========================================================
    # GENERIC GROUP
    # =========================================================

    else:

        groups = sorted(
            df[group_column]
            .dropna()
            .unique()
        )

        if val_groups is None:

            n_val = max(
                1,
                round(len(groups) * val_fraction)
            )

            val_groups = set(
                rng.choice(
                    groups,
                    size=n_val,
                    replace=False,
                )
            )

        else:
            val_groups = set(val_groups)

        val_mask = df[group_column].isin(val_groups)

        df["split"] = "train"
        df.loc[val_mask, "split"] = "val"

        print(
            f"Validation {group_column}:"
        )
        print(sorted(val_groups))

        print(
            f"\nValidation images: "
            f"{val_mask.sum()} / {len(df)} "
            f"({val_mask.mean():.1%})"
        )

    # =========================================================
    # OUTPUT
    # =========================================================

    val_set = set(
        df.loc[
            df["split"] == "val",
            "full_path"
        ]
    )

    train_set = set(
        df.loc[
            df["split"] == "train",
            "full_path"
        ]
    )

    return val_set, train_set, df


def apply_split(src, dst, val_samples, train_samples):
    src = Path(src)
    dst = Path(dst)

    if not src.is_dir():
        raise ValueError(f"Source directory '{src}' does not exist.")

    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    image_prefix = "data"
    mask_prefix = "labels"

    for set_name, set in zip(["train", "val"], [train_samples, val_samples]):
        
        # Copy the files
        for sample in tqdm(set, desc=f"Copying {set_name} samples"):
            image_path = Path(src) / Path(image_prefix) / Path(sample)
            if not image_path.exists():
                raise Exception("File {} does not exist".format(str(image_path)))

            mask_path = Path(src) / Path(mask_prefix) / Path(sample)
            if not mask_path.exists():
                raise Exception("File {} does not exist".format(str(mask_path)))

            # Copy to given location train set
            dst_img = Path(dst) / Path(set_name) / Path(image_prefix) / Path(sample)
            dst_mask = Path(dst) / Path(set_name) / Path(mask_prefix) / Path(sample)

            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_mask.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(str(image_path), str(dst_img))
            shutil.copy(str(mask_path), str(dst_mask))

      
# # split SYMPTOMS dataset
# src_directory = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/04_DL_datasets_updates/symptoms")
# metadata = pd.read_csv(src_directory / "meta" / "merged_metadata_batch1-9_11-14.csv", dtype={"plot": "Int64"})

# val_set, train_set, metadata_split = split_dataset(
#     metadata,
#     val_fraction=0.2,
#     strategy="group",
#     group_column=["genotype_name"],
#     val_groups=None,
# )

# split FOCUS dataset
src_directory = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/04_DL_datasets_updates/focus")
metadata = pd.read_csv(src_directory / "meta" / "merged_metadata.csv", dtype={"plot": "Int64"})

val_set, train_set, metadata_split = split_dataset(
    metadata,
    val_fraction=0.2,
    strategy="random",
    group_column=None,
    val_groups=None,
)

val_samples = [Path(sample).stem + ".png" for sample in val_set]
train_samples = [Path(sample).stem + ".png" for sample in train_set]

export_src_root = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/01_DL_Datasets/05_CANOPY_PROXIMAL_FOCUS/dataset_src/segmentations_export")
split_dst_root = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/01_DL_Datasets/20_Datasets_train/Focus")

apply_split(export_src_root, split_dst_root, val_samples, train_samples)