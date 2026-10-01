import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MATCH_FILE = os.path.join(
    BASE_DIR,
    "results",
    "aqi_mapping",
    "aqi_vit_embedding_matches.csv"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "results",
    "vit_embeddings"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "aqi_regression"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CREATING AQI + ViT EMBEDDING DATASET")
print("=" * 70)


# ============================================================
# LOAD SUCCESSFUL AQI ↔ EMBEDDING MATCHES
# ============================================================

matches = pd.read_csv(MATCH_FILE)

print(
    f"\nMatched AQI records: {len(matches)}"
)

print(
    "\nColumns:"
)

print(
    matches.columns.tolist()
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embeddings_by_split = {}

for split in ["train", "val", "test"]:

    embedding_file = os.path.join(
        EMBEDDING_DIR,
        f"{split}_embeddings.npy"
    )

    embeddings_by_split[split] = np.load(
        embedding_file
    )

    print(
        f"\n{split} embeddings: "
        f"{embeddings_by_split[split].shape}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

metadata_by_split = {}

for split in ["train", "val", "test"]:

    metadata_file = os.path.join(
        EMBEDDING_DIR,
        f"{split}_embedding_metadata.csv"
    )

    metadata_by_split[split] = pd.read_csv(
        metadata_file
    )


# ============================================================
# BUILD EXACT PATH LOOKUP
# ============================================================

embedding_lookup = {}


for split in ["train", "val", "test"]:

    metadata = metadata_by_split[split]
    embeddings = embeddings_by_split[split]

    for index, row in metadata.iterrows():

        image_path = os.path.normcase(
            os.path.normpath(
                str(row["image_path"])
            )
        )

        embedding_lookup[
            image_path
        ] = embeddings[index]


# ============================================================
# MATCH USING THE EXACT PATH
# ============================================================

final_records = []
final_embeddings = []

unmatched = []


for _, row in matches.iterrows():

    embedding_path = os.path.normcase(
        os.path.normpath(
            str(
                row["embedding_metadata_path"]
            )
        )
    )

    if embedding_path in embedding_lookup:

        final_records.append({
            "filename": row["filename"],
            "phyllotaxy": row["phyllotaxy"],
            "AQI": float(row["aqi"]),
            "split": row["split"],
            "embedding_path": row[
                "embedding_metadata_path"
            ]
        })

        final_embeddings.append(
            embedding_lookup[
                embedding_path
            ]
        )

    else:

        unmatched.append({
            "filename": row["filename"],
            "phyllotaxy": row["phyllotaxy"],
            "AQI": row["aqi"],
            "split": row["split"],
            "embedding_path": row[
                "embedding_metadata_path"
            ]
        })


# ============================================================
# CONVERT EMBEDDINGS
# ============================================================

if len(final_embeddings) > 0:

    final_embeddings = np.asarray(
        final_embeddings,
        dtype=np.float32
    )

else:

    final_embeddings = np.empty(
        (0, 768),
        dtype=np.float32
    )


# ============================================================
# METADATA DATAFRAME
# ============================================================

final_metadata = pd.DataFrame(
    final_records
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"\nFinal samples: "
    f"{len(final_metadata)}"
)

print(
    f"Embedding shape: "
    f"{final_embeddings.shape}"
)

print(
    f"Unmatched: "
    f"{len(unmatched)}"
)


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

if len(final_metadata) > 0:

    print(
        "\nSamples by split:"
    )

    print(
        final_metadata[
            "split"
        ].value_counts()
    )

    print(
        "\nAQI statistics:"
    )

    print(
        final_metadata[
            "AQI"
        ].describe()
    )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

embedding_output = os.path.join(
    OUTPUT_DIR,
    "aqi_vit_embeddings.npy"
)

np.save(
    embedding_output,
    final_embeddings
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_output = os.path.join(
    OUTPUT_DIR,
    "aqi_vit_metadata.csv"
)

final_metadata.to_csv(
    metadata_output,
    index=False
)


# ============================================================
# SAVE COMPLETE DATASET
# ============================================================

npz_output = os.path.join(
    OUTPUT_DIR,
    "aqi_vit_dataset.npz"
)

np.savez(
    npz_output,
    embeddings=final_embeddings,
    aqi=final_metadata[
        "AQI"
    ].to_numpy(
        dtype=np.float32
    ),
    filenames=final_metadata[
        "filename"
    ].to_numpy(
        dtype=str
    ),
    splits=final_metadata[
        "split"
    ].to_numpy(
        dtype=str
    )
)


# ============================================================
# SAVE UNMATCHED RECORDS
# ============================================================

if len(unmatched) > 0:

    unmatched_output = os.path.join(
        OUTPUT_DIR,
        "final_unmatched.csv"
    )

    pd.DataFrame(
        unmatched
    ).to_csv(
        unmatched_output,
        index=False
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\nSaved:"
)

print(
    embedding_output
)

print(
    metadata_output
)

print(
    npz_output
)

print(
    "\n" + "=" * 70
)

print(
    "AQI + ViT DATASET CREATION COMPLETE"
)

print(
    "=" * 70
)