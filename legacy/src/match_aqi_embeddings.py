import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AQI_FILE = os.path.join(
    BASE_DIR,
    "results",
    "aqi_mapping",
    "clean_image_aqi_mapping.csv"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "results",
    "vit_embeddings"
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
# LOAD CLEAN AQI DATA
# ============================================================

print("=" * 70)
print("AQI ↔ ViT EMBEDDING MATCHING")
print("=" * 70)

aqi_df = pd.read_csv(AQI_FILE)

print(
    f"\nClean AQI records: {len(aqi_df)}"
)


# ============================================================
# LOAD EMBEDDING METADATA
# ============================================================

metadata_files = [
    "train_embedding_metadata.csv",
    "val_embedding_metadata.csv",
    "test_embedding_metadata.csv"
]

all_metadata = []

for filename in metadata_files:

    filepath = os.path.join(
        EMBEDDING_DIR,
        filename
    )

    df = pd.read_csv(filepath)

    # Determine split from filename
    if filename.startswith("train"):
        df["split"] = "train"

    elif filename.startswith("val"):
        df["split"] = "val"

    else:
        df["split"] = "test"

    all_metadata.append(df)


embedding_df = pd.concat(
    all_metadata,
    ignore_index=True
)

print(
    f"Total embedding records: {len(embedding_df)}"
)


# ============================================================
# EXTRACT BASENAME
# ============================================================

aqi_df["filename"] = aqi_df[
    "filename_only"
].astype(str).str.lower()


embedding_df["filename"] = (
    embedding_df["image_path"]
    .astype(str)
    .apply(
        lambda x: os.path.basename(
            x.replace("\\", "/")
        ).lower()
    )
)


# ============================================================
# NORMALIZE CLASS NAMES
# ============================================================

aqi_df["class_normalized"] = (
    aqi_df["folder"]
    .astype(str)
    .str.lower()
    .str.replace(" leaf", "", regex=False)
)

embedding_df["class_normalized"] = (
    embedding_df["phyllotaxy"]
    .astype(str)
    .str.lower()
)


# ============================================================
# MATCH USING CLASS + FILENAME
# ============================================================

matched = []

unmatched = []

for _, row in aqi_df.iterrows():

    matches = embedding_df[
        (
            embedding_df["filename"]
            == row["filename"]
        )
        &
        (
            embedding_df["class_normalized"]
            == row["class_normalized"]
        )
    ]

    if len(matches) == 1:

        match = matches.iloc[0]

        matched.append({
            "filename": row["filename_only"],
            "aqi": row["AQI"],
            "phyllotaxy": row["folder"],
            "split": match["split"],
            "embedding_metadata_path": match["image_path"]
        })

    elif len(matches) == 0:

        unmatched.append({
            "filename": row["filename_only"],
            "aqi": row["AQI"],
            "phyllotaxy": row["folder"]
        })

    else:

        print(
            f"WARNING: Multiple embedding matches "
            f"for {row['filename_only']}"
        )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

matched_df = pd.DataFrame(
    matched
)

unmatched_df = pd.DataFrame(
    unmatched
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING RESULTS")
print("=" * 70)

print(
    f"\nAQI records:        {len(aqi_df)}"
)

print(
    f"Matched embeddings: {len(matched_df)}"
)

print(
    f"Unmatched:          {len(unmatched_df)}"
)


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

if len(matched_df) > 0:

    print(
        "\nMatched records by split:"
    )

    print(
        matched_df["split"].value_counts()
    )

    print(
        "\nMatched records by phyllotaxy:"
    )

    print(
        matched_df["phyllotaxy"].value_counts()
    )

    print(
        "\nAQI statistics:"
    )

    print(
        matched_df["aqi"].describe()
    )


# ============================================================
# UNMATCHED RECORDS
# ============================================================

if len(unmatched_df) > 0:

    print(
        "\nUnmatched AQI images:"
    )

    print(
        unmatched_df.to_string(
            index=False
        )
    )


# ============================================================
# SAVE MATCHED DATA
# ============================================================

matched_file = os.path.join(
    OUTPUT_DIR,
    "aqi_vit_embedding_matches.csv"
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
    "aqi_vit_embedding_unmatched.csv"
)

unmatched_df.to_csv(
    unmatched_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\nResults saved to:"
)

print(
    matched_file
)

print(
    unmatched_file
)

print("\n" + "=" * 70)
print("MATCHING COMPLETE")
print("=" * 70)