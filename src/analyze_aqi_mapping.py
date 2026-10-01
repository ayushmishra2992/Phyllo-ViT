import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

AQI_FILE = os.path.join(
    RAW_DIR,
    "data AQI.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "aqi_mapping"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("IMAGE ↔ AQI MAPPING ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD AQI EXCEL FILE
# ============================================================

print("\nLoading AQI data...")

df = pd.read_excel(AQI_FILE)

print(f"Total AQI records: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# EXTRACT FILENAME AND FOLDER
# ============================================================

df["filename_only"] = df["Filename"].astype(str).apply(
    lambda x: os.path.basename(
        x.replace("\\", "/")
    )
)

df["folder"] = df["Filename"].astype(str).apply(
    lambda x: os.path.basename(
        os.path.dirname(
            x.replace("\\", "/")
        )
    )
)


# ============================================================
# AQI DATA SUMMARY
# ============================================================

print("\nAQI records by folder/class:")

print(
    df["folder"].value_counts()
)


print("\nAQI statistics:")

print(
    df["AQI"].describe()
)


# ============================================================
# RAW DATASET FOLDERS
# ============================================================

raw_classes = {
    "Alternate leaf": os.path.join(
        RAW_DIR,
        "alternate",
        "Alternate leaf"
    ),

    "Opposite leaf": os.path.join(
        RAW_DIR,
        "opposite",
        "Opposite leaf"
    ),

    "Whorled leaf": os.path.join(
        RAW_DIR,
        "whorled",
        "Whorled leaf"
    )
}


# ============================================================
# BUILD RAW IMAGE INDEX
# ============================================================

print(
    "\nChecking whether AQI filenames exist "
    "in the corresponding raw class folder..."
)

all_raw_files = {}

for class_name, folder in raw_classes.items():

    print(
        f"\n{class_name}:"
    )

    if not os.path.exists(folder):

        print(
            f"  ❌ Folder not found: {folder}"
        )

        continue

    files = os.listdir(folder)

    image_count = 0

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
            continue

        image_count += 1

        # Match using BOTH:
        # class/folder + filename
        key = (
            class_name.lower(),
            filename.lower()
        )

        all_raw_files[key] = {
            "filename": filename,
            "class": class_name,
            "path": path
        }

    print(
        f"  Images found: {image_count}"
    )


# ============================================================
# MATCH AQI RECORDS TO RAW IMAGES
# ============================================================

matched = []
unmatched = []

for _, row in df.iterrows():

    filename = row["filename_only"]
    excel_folder = row["folder"]

    key = (
        excel_folder.lower(),
        filename.lower()
    )

    if key in all_raw_files:

        raw_info = all_raw_files[key]

        matched.append({
            "filename": filename,
            "aqi": row["AQI"],
            "class": raw_info["class"],
            "raw_path": raw_info["path"]
        })

    else:

        unmatched.append({
            "filename": filename,
            "aqi": row["AQI"],
            "folder_from_excel": excel_folder
        })


# ============================================================
# CONVERT TO DATAFRAMES
# ============================================================

matched_df = pd.DataFrame(
    matched
)

unmatched_df = pd.DataFrame(
    unmatched
)


# ============================================================
# MATCHING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING RESULTS")
print("=" * 70)

print(
    f"\nAQI records:       {len(df)}"
)

print(
    f"Matched to raw:    {len(matched_df)}"
)

print(
    f"Not matched:       {len(unmatched_df)}"
)


# ============================================================
# MATCHED CLASS DISTRIBUTION
# ============================================================

if len(matched_df) > 0:

    print(
        "\nMatched records by class:"
    )

    print(
        matched_df["class"].value_counts()
    )


# ============================================================
# MATCHED AQI STATISTICS
# ============================================================

if len(matched_df) > 0:

    print(
        "\nMatched AQI statistics by class:"
    )

    print(
        matched_df
        .groupby("class")["aqi"]
        .describe()
    )


# ============================================================
# CHECK FOR DUPLICATE IMAGE + AQI RECORDS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DUPLICATE / CONFLICT CHECK"
)

print(
    "=" * 70
)


# Same class + same filename appearing multiple times
duplicate_rows = (
    matched_df[
        matched_df.duplicated(
            subset=["class", "filename"],
            keep=False
        )
    ]
    .sort_values(
        ["class", "filename"]
    )
)


if len(duplicate_rows) > 0:

    print(
        "\n⚠️ Duplicate filename records found:"
    )

    print(
        duplicate_rows.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo duplicate filename records found."
    )


# ============================================================
# CHECK FOR CONFLICTING AQI VALUES
# ============================================================

if len(duplicate_rows) > 0:

    conflict_groups = (
        duplicate_rows
        .groupby(
            ["class", "filename"]
        )["aqi"]
        .nunique()
    )

    conflicts = conflict_groups[
        conflict_groups > 1
    ]

    if len(conflicts) > 0:

        print(
            "\n⚠️ AQI conflicts found:"
        )

        for index, count in conflicts.items():

            class_name, filename = index

            values = (
                duplicate_rows[
                    (duplicate_rows["class"] == class_name)
                    &
                    (duplicate_rows["filename"] == filename)
                ]["aqi"]
                .tolist()
            )

            print(
                f"  {class_name} / {filename}: "
                f"AQI values = {values}"
            )

    else:

        print(
            "\nNo conflicting AQI values found."
        )


# ============================================================
# UNMATCHED RECORDS
# ============================================================

if len(unmatched_df) > 0:

    print(
        "\n" + "=" * 70
    )

    print(
        "UNMATCHED RECORDS"
    )

    print(
        "=" * 70
    )

    print(
        unmatched_df.to_string(
            index=False
        )
    )

else:

    print(
        "\nAll AQI records matched to raw images."
    )


# ============================================================
# SAVE MATCHED DATA
# ============================================================

matched_file = os.path.join(
    OUTPUT_DIR,
    "matched_aqi_images.csv"
)

matched_df.to_csv(
    matched_file,
    index=False
)


# ============================================================
# SAVE UNMATCHED DATA
# ============================================================

unmatched_file = os.path.join(
    OUTPUT_DIR,
    "unmatched_aqi_images.csv"
)

unmatched_df.to_csv(
    unmatched_file,
    index=False
)


# ============================================================
# SAVE DUPLICATE RECORDS
# ============================================================

duplicate_file = os.path.join(
    OUTPUT_DIR,
    "duplicate_aqi_records.csv"
)

duplicate_rows.to_csv(
    duplicate_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nMatched records:   {len(matched_df)}"
)

print(
    f"Unmatched records: {len(unmatched_df)}"
)

print(
    f"Duplicate records: {len(duplicate_rows)}"
)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_DIR
)