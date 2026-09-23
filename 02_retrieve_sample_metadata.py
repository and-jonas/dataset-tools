
from pathlib import Path
import re
import csv
import exifread
import pandas as pd
from IPython.display import display

src_directory = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/04_DL_datasets_updates/symptoms")

# ------------------------------------------------------------------------------------------
# Get Meta information from source image paths and write to CSV
# ------------------------------------------------------------------------------------------

# guides for finding relevant meta information
# TODO Update as needed when new batches are included
exp_IDs = ["CHWW001", "PreDiMix"]
loc_IDs = ["Uitikon", "Eschikon", "02_CHWW001"]

# select only batches 1-9
dirs = [d for d in src_directory.iterdir() if d.is_dir()]
dirs = [p for p in dirs if p.name in {f"batch{i}" for i in range(1, 10)}]

# helper for datetime extraction from source image path
def extract_datetime_original(file_path: Path) -> str:
    with file_path.open("rb") as f:
        tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
    dtime_tag = tags.get("EXIF DateTimeOriginal")
    if not dtime_tag:
        return file_path.stem
    dtime = str(dtime_tag.values)
    dtime = dtime.replace(":", "")
    return dtime.replace(" ", "_")

rows = []
for d in dirs:
    meta_file = d / "img" / "source.txt"
    if not meta_file.exists():
        print(f"Missing meta file: {meta_file}")
        continue

    with open(meta_file, "r", encoding="utf-8") as f:
        source_lines = [line.strip() for line in f if line.strip()]

    # Get stems of available patches
    patches_dir = d / "patches"
    if patches_dir.exists() and patches_dir.is_dir():
        patches = {
            p.stem
            for p in patches_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
    else:
        patches = set()

    # Keep only lines that have a corresponding patch
    if patches:
        source_lines = [line for line in source_lines if Path(line).stem in patches]

    for line in source_lines:
        # Experiment ID
        exp_id = next((x for x in exp_IDs if x in line), None)
        if exp_id is None:
            raise ValueError(f"Line '{line}' does not contain a valid experiment ID.")

        # Location ID
        loc_id = next((x for x in loc_IDs if x in line), None)
        if loc_id is None:
            raise ValueError(f"Line '{line}' does not contain a valid location ID.")
        if loc_id == "02_CHWW001":
            loc_id = "Changins"

        # Harvest Year: find any 4-digit pattern that looks like a year
        year_matches = re.findall(r"\b(20\d{2})", line)
        if not year_matches:    
            raise ValueError(f"No valid harvest year found in line: '{line}'")  
        harvest_year = year_matches[0]

        # Time of Day
        path = Path(line.replace("\\", "/").strip('"'))
        time_of_day = extract_datetime_original(path).split("_")[1]

        if not path.is_absolute():
            # resolve relative paths with respect to the batch directory
            try:
                path = (d / path).resolve()
            except Exception:
                path = (d / path)

        # Date: find any 8-digit pattern that looks like YYYYMMDD
        date_matches = re.findall(r"\b(20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))", line)
        if not date_matches:
            raise ValueError(f"No valid date found in line: '{line}'")
        date = date_matches[0]

        # Plot: find a number of varying length ending in 4 digits
        final_dir = path.parent.name
        plot_matches = re.findall(r"\b[A-Za-z0-9]+\d{4}\b", final_dir)

        if not plot_matches:
            if 'O' in final_dir or 'o' in final_dir:
                alternative = final_dir.replace('O', '0').replace('o', '0')
                print(f"Possible 0/O confusion: {final_dir}  ->  {alternative}")
                plot_matches = re.findall(r"\b[A-Za-z0-9]+\d{4}\b", alternative)

        if not plot_matches:
            raise ValueError(f"No numeric plot ID found in final directory: '{final_dir}'")

        plot = plot_matches[-1][-4:]

        # image name
        image_name = path.name

        # gather output
        rows.append({
            "full_path": str(path),
            "experiment_name": exp_id,
            "location_name": loc_id,
            "date": date,
            "time_of_day": time_of_day,
            "harvest_year": harvest_year,
            "plot": plot,
            "image_name": image_name,
        })

    print(f'number of files: {len(source_lines)} in {d.name}')


# Write to CSV
output_file = src_directory / "meta" / "source_filtered_batch1-9.csv"
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["full_path", "experiment_name", "location_name", "date", "time_of_day", "harvest_year", "plot", "image_name"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote {len(rows)} rows to {output_file}")


# ------------------------------------------------------------------------------------------
# Get additional meta information from experimental designs
# ------------------------------------------------------------------------------------------

design_CHWW001 = pd.read_csv(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/A_Experiments/CHWW001/design_complete_10_4.csv"
)
design_PrediMix_Uitikon = pd.read_csv(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/A_Experiments/PreDiMix/Designs/Design_CH1306392026_FIELDBOOK_PREDIMIX_Uitikon.csv"
)
design_PreDiMix_Eschikon = pd.read_csv(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/A_Experiments/PreDiMix/Designs/Design_CH0304112026_FIELDBOOK_PREDIMIX_Eschikon.csv"
)

# no treatments were finally applied in CHWW001
design_CHWW001["treatment_name"] = "0"

# select relevant columns and add missing columns where relevant
# Changins CHWW001
d_CHWW001 = design_CHWW001[[
    "exp_UID", "plot_UID", "plot", "harvest_year", "genotype_name", "genotype_UID", "treatment_name"
    ]].rename(columns={"exp_UID": "experiment_UID"})
d_CHWW001["experiment_name"] = "CHWW001"
d_CHWW001["genotype_name1"] = d_CHWW001["genotype_name"] 
d_CHWW001["genotype_UID1"] = d_CHWW001["genotype_UID"] 
d_CHWW001["zip_code"] = 1260
d_CHWW001["location_name"] = "changins"

# Uitikon PreDiMix
d_PrediMix_Uitikon = design_PrediMix_Uitikon[[
    "experiment_UID", "plot_UID", "plot", "harvest_year", 
    "genotype_name", "genotype_name1", "genotype_name2", 
    "gen_id", "gen_id1", "gen_id2", "location_name", "zip_code"
    ]].rename(columns={"gen_id": "genotype_UID", "gen_id1": "genotype_UID1", "gen_id2": "genotype_UID2"})
d_PrediMix_Uitikon["experiment_name"] = "PreDiMix"
d_PrediMix_Uitikon["treatment_name"] = "0"

# Eschikon PreDiMix
d_PreDiMix_Eschikon = design_PreDiMix_Eschikon[[
    "experiment_UID", "plot_UID", "plot", "harvest_year", 
    "genotype_name", "genotype_name1", "genotype_name2", 
    "gen_id", "gen_id1", "gen_id2", "location_name", "zip_code"
    ]].rename(columns={"gen_id": "genotype_UID", "gen_id1": "genotype_UID1", "gen_id2": "genotype_UID2"})
d_PreDiMix_Eschikon["experiment_name"] = "PreDiMix"
d_PreDiMix_Eschikon["treatment_name"] = "0"

# bind all columns
all = pd.concat([d_CHWW001, d_PrediMix_Uitikon, d_PreDiMix_Eschikon], axis=0, ignore_index=True, sort=False)

# adjust data types
dtypes = {
    "experiment_UID": "string",
    "experiment_name": "string",
    "plot_UID": "string",
    "plot": "Int64",
    "harvest_year": "Int64",
    "genotype_name": "string",
    "experiment_name": "string",
    "genotype_name1": "string",
    "location_name": "string",
    "zip_code": "Int64",
    "genotype_UID1": "Int64",
    "genotype_name2": "string",
    "genotype_UID2": "Int64",
    "treatment_name": "string",
}
all = all.astype(dtypes)
all.to_csv(src_directory / "meta" / "designs.csv", index=False)

# ------------------------------------------------------------------------------------------
# Merge source metadata with experimental designs
# ------------------------------------------------------------------------------------------

source = pd.read_csv(src_directory / "meta" / "source_filtered_batch1-9.csv", 
                     dtype={"plot": "Int64"})
source["location_name"] = source["location_name"].str.strip().str.casefold()
result = source.merge(all, how="left", 
                      on=["plot", "harvest_year", "experiment_name", "location_name"])
result.to_csv(src_directory / "meta" / "merged_metadata_batch1-9.csv", index=False)