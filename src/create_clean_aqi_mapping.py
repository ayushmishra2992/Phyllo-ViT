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
# LOAD AQI DATA
# ============================================================

df = pd.read_excel(AQI_FILE)

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
# WHORLED IMAGE DIRECTORY
# ============================================================

WHORLED_DIR = os.path.join(
    RAW_DIR,
    "whorled",
    "Whorled leaf"
)


# ============================================================
# REMOVE MISSING IMAGE
# ============================================================

df["image_exists"] = df["filename_only"].apply(
    lambda x: os.path.exists(
        os.path.join(
            WHORLED_DIR,
            x
        )
    )
)


# ============================================================
# REMOVE CONFLICTING FILENAMES
# ============================================================

# A filename is considered conflicting if it has
# more than one different AQI value.

aqi_counts = (
    df.groupby("filename_only")["AQI"]
    .nunique()
)

conflicting_filenames = (
    aqi_counts[
        aqi_counts > 1
    ]
    .index
    .tolist()
)


print("=" * 70)
print("CREATING CLEAN IMAGE ↔ AQI DATASET")
print("=" * 70)

print(
    f"\nOriginal AQI records: {len(df)}"
)

print(
    f"Missing images: "
    f"{(~df['image_exists']).sum()}"
)

print(
    f"Conflicting filenames: "
    f"{len(conflicting_filenames)}"
)

print(
    f"Conflicting filename(s): "
    f"{conflicting_filenames}"
)


# ============================================================
# KEEP ONLY VALID RECORDS
# ============================================================

clean_df = df[
    df["image_exists"]
    &
    ~df["filename_only"].isin(
        conflicting_filenames
    )
].copy()


# ============================================================
# ADD FULL IMAGE PATH
# ============================================================

clean_df["image_path"] = clean_df[
    "filename_only"
].apply(
    lambda x: os.path.join(
        WHORLED_DIR,
        x
    )
)


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

clean_df = clean_df[
    [
        "filename_only",
        "image_path",
        "folder",
        "AQI"
    ]
]


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_clean = clean_df[
    clean_df.duplicated(
        subset=["filename_only"],
        keep=False
    )
]

print(
    f"\nDuplicate records remaining: "
    f"{len(duplicate_clean)}"
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "clean_image_aqi_mapping.csv"
)

clean_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN AQI DATASET CREATED")
print("=" * 70)

print(
    f"\nValid image-AQI pairs: "
    f"{len(clean_df)}"
)

print(
    f"Minimum AQI: "
    f"{clean_df['AQI'].min()}"
)

print(
    f"Maximum AQI: "
    f"{clean_df['AQI'].max()}"
)

print(
    f"Mean AQI: "
    f"{clean_df['AQI'].mean():.2f}"
)

print(
    f"\nSaved to:"
)

print(
    output_file
)

print("\n" + "=" * 70)