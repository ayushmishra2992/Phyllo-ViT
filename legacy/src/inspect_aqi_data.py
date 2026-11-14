import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

FILES = [
    "AQI data LT.xlsx",
    "AQI data.xlsx",
    "data AQI.xlsx",
    "NISE Gwal Pahari,Gurugram-IMD.xlsx",
]

print("=" * 80)
print("AQI / AIR-QUALITY DATASET INSPECTION")
print("=" * 80)

for filename in FILES:

    filepath = os.path.join(RAW_DIR, filename)

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    if not os.path.exists(filepath):
        print("❌ FILE NOT FOUND")
        continue

    try:
        excel = pd.ExcelFile(filepath)

        print("\nSheets:")
        for sheet in excel.sheet_names:
            print(f"  - {sheet}")

        for sheet in excel.sheet_names:

            print("\n" + "-" * 80)
            print(f"SHEET: {sheet}")
            print("-" * 80)

            df = pd.read_excel(
                filepath,
                sheet_name=sheet
            )

            print(f"Rows: {df.shape[0]}")
            print(f"Columns: {df.shape[1]}")

            print("\nColumn names:")
            for column in df.columns:
                print(f"  - {column}")

            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))

            print("\nData types:")
            print(df.dtypes.to_string())

            print("\nMissing values:")
            missing = df.isnull().sum()
            print(missing[missing > 0].to_string())

    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)