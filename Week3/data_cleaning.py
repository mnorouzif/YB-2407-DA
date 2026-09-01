import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. LOAD THE DATASET
# ============================================================

file_path = "Week3/Sample_dataset.csv"

df = pd.read_csv(file_path)

print("Original Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# ============================================================
# 3. CHECK DUPLICATES
# ============================================================

print("\nNumber of Exact Duplicate Rows:")
print(df.duplicated().sum())

print("\nDuplicate Rows:")
print(df[df.duplicated(keep=False)])


# Check duplicate IDs
print("\nDuplicate IDs:")
print(df[df["ID"].duplicated(keep=False) & df["ID"].notna()])


# ============================================================
# 4. CREATE A COPY FOR CLEANING
# ============================================================

clean = df.copy()


# ============================================================
# 5. CLEAN ID
# ============================================================

clean["ID"] = pd.to_numeric(
    clean["ID"],
    errors="coerce"
).astype("Int64")


# ============================================================
# 6. CLEAN AGE
# ============================================================

# Convert normal numeric values
clean["Age"] = pd.to_numeric(
    clean["Age"],
    errors="coerce"
)

# Recover text value "thirty-eight"
clean.loc[
    clean["Age"].isna()
    & df["Age"].astype(str).str.lower().eq("thirty-eight"),
    "Age"
] = 38


# ============================================================
# 7. CLEAN NET WORTH
# ============================================================

clean["Net worth"] = pd.to_numeric(
    clean["Net worth"]
    .astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce"
)


# ============================================================
# 8. CLEAN SALARY
# ============================================================

salary_text = (
    clean["Salary"]
    .astype(str)
    .str.lower()
    .str.replace(",", "", regex=False)
    .str.strip()
)

clean["Salary"] = pd.to_numeric(
    salary_text,
    errors="coerce"
)

# Recover written value
clean.loc[
    salary_text.eq("sixty five thousand"),
    "Salary"
] = 65000


# ============================================================
# 9. STANDARDISE COUNTRY VALUES
# ============================================================

# Convert AU to AUS
clean["Country"] = clean["Country"].replace({
    "AU": "AUS"
})


# ============================================================
# 10. CLEAN JOIN DATE
# ============================================================

clean["Join Date"] = pd.to_datetime(
    clean["Join Date"],
    dayfirst=True,
    errors="coerce"
)

# Recover invalid date:
# 2019-13-01 is interpreted here as 13 January 2019
bad_date = df["Join Date"].astype(str).eq("2019-13-01")

clean.loc[
    bad_date,
    "Join Date"
] = pd.Timestamp("2019-01-13")


# ============================================================
# 11. HANDLE DUPLICATE BOB RECORDS
# ============================================================

duplicate_bob = (
    clean["ID"].eq(2)
    & clean["Name"].eq("Bob")
)

bob_rows = clean.loc[duplicate_bob]

print("\nBob duplicate records:")
print(bob_rows)

# If more than one Bob record exists,
# combine their non-missing information.
if len(bob_rows) > 1:

    bob = bob_rows.iloc[0].copy()

    for column in clean.columns:

        values = bob_rows[column].dropna()

        if pd.isna(bob[column]) and len(values) > 0:
            bob[column] = values.iloc[0]

    # Remove duplicate Bob records
    clean = clean.loc[~duplicate_bob].copy()

    # Add the merged Bob record
    clean = pd.concat(
        [clean, pd.DataFrame([bob])],
        ignore_index=True
    )


# ============================================================
# 12. SORT DATA BY ID
# ============================================================

clean = clean.sort_values(
    "ID",
    na_position="last"
).reset_index(drop=True)


# ============================================================
# 13. HANDLE MISSING NUMERIC VALUES
# ============================================================

numeric_columns = [
    "Age",
    "Net worth",
    "Salary"
]

imputation_values = {}

for column in numeric_columns:

    median_value = clean[column].median()

    imputation_values[column] = median_value

    clean[column] = clean[column].fillna(
        median_value
    )


# Display imputation values
print("\nMedian values used for imputation:")

for column, value in imputation_values.items():
    print(column, "=", value)


# ============================================================
# 14. HANDLE MISSING COUNTRY
# ============================================================

country_mode = clean["Country"].mode()[0]

clean["Country"] = clean["Country"].fillna(
    country_mode
)

print("\nCountry value used for missing data:")
print(country_mode)


# ============================================================
# 15. KEEP UNKNOWN NAME AND ID AS MISSING
# ============================================================

# We do NOT invent a person's name or ID.
# Therefore, missing Name and ID remain missing.


# ============================================================
# 16. FORMAT DATA TYPES
# ============================================================

clean["ID"] = clean["ID"].astype("Int64")

clean["Age"] = (
    clean["Age"]
    .round()
    .astype("Int64")
)

clean["Net worth"] = (
    clean["Net worth"]
    .round()
    .astype("Int64")
)

clean["Salary"] = (
    clean["Salary"]
    .round()
    .astype("Int64")
)


# Format date
clean["Join Date"] = clean["Join Date"].dt.strftime(
    "%d/%m/%Y"
)


# ============================================================
# 17. DISPLAY CLEAN DATASET
# ============================================================

print("\n===================================")
print("CLEANED DATASET")
print("===================================")

print(clean.to_string(index=False))


# ============================================================
# 18. CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\nMissing Values After Cleaning:")
print(clean.isnull().sum())

print(
    "\nTotal Missing Values After Cleaning:",
    clean.isnull().sum().sum()
)


# ============================================================
# 19. COMPARE BEFORE AND AFTER
# ============================================================

print("\n===================================")
print("BEFORE vs AFTER")
print("===================================")

print(
    "Original rows:",
    len(df)
)

print(
    "Cleaned rows:",
    len(clean)
)

print(
    "Original missing values:",
    df.isnull().sum().sum()
)

print(
    "Remaining missing values:",
    clean.isnull().sum().sum()
)

print(
    "Exact duplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 20. BASIC DATA ANALYSIS
# ============================================================

print("\n===================================")
print("BASIC DATA ANALYSIS")
print("===================================")

print("\nAverage Age:")
print(clean["Age"].mean())

print("\nAverage Salary:")
print(clean["Salary"].mean())

print("\nMaximum Salary:")
print(clean["Salary"].max())

print("\nMinimum Salary:")
print(clean["Salary"].min())

print("\nAverage Net Worth:")
print(clean["Net worth"].mean())

print("\nEmployees by Country:")
print(clean["Country"].value_counts())


# ============================================================
# 21. CREATE OUTPUT FOLDER
# ============================================================

output_folder = Path("data_cleaning_output")

output_folder.mkdir(
    exist_ok=True
)


# ============================================================
# 22. SAVE CLEANED DATA
# ============================================================

clean.to_csv(
    output_folder / "Sample_dataset_cleaned.csv",
    index=False
)

print(
    "\nCleaned dataset saved as:",
    output_folder / "Sample_dataset_cleaned.csv"
)


# ============================================================
# 23. VISUALISATION 1
#     MISSING VALUES BEFORE CLEANING
# ============================================================

missing_before = df.isnull().sum()

missing_before = missing_before[
    missing_before > 0
]

plt.figure(figsize=(8, 5))

missing_before.plot(
    kind="bar"
)

plt.title(
    "Missing Values Before Cleaning"
)

plt.xlabel("Column")

plt.ylabel(
    "Number of Missing Values"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_folder / "01_missing_values_before.png",
    dpi=160
)

plt.show()


# ============================================================
# 24. VISUALISATION 2
#     MISSING VALUES AFTER CLEANING
# ============================================================

missing_after = clean.isnull().sum()

missing_after = missing_after[
    missing_after > 0
]

plt.figure(figsize=(8, 5))

missing_after.plot(
    kind="bar"
)

plt.title(
    "Remaining Missing Values After Cleaning"
)

plt.xlabel("Column")

plt.ylabel(
    "Number of Missing Values"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_folder / "02_missing_values_after.png",
    dpi=160
)

plt.show()


# ============================================================
# 25. VISUALISATION 3
#     SALARY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

clean["Salary"].astype(float).plot(
    kind="hist",
    bins=6
)

plt.title(
    "Salary Distribution"
)

plt.xlabel(
    "Salary"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    output_folder / "03_salary_distribution.png",
    dpi=160
)

plt.show()


# ============================================================
# 26. VISUALISATION 4
#     NET WORTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

clean["Net worth"].astype(float).plot(
    kind="hist",
    bins=6
)

plt.title(
    "Net Worth Distribution"
)

plt.xlabel(
    "Net Worth"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    output_folder / "04_net_worth_distribution.png",
    dpi=160
)

plt.show()


# ============================================================
# 27. VISUALISATION 5
#     SALARY VS NET WORTH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    clean["Salary"].astype(float),
    clean["Net worth"].astype(float)
)

plt.title(
    "Salary vs Net Worth"
)

plt.xlabel(
    "Salary"
)

plt.ylabel(
    "Net Worth"
)

plt.tight_layout()

plt.savefig(
    output_folder / "05_salary_vs_net_worth.png",
    dpi=160
)

plt.show()


# ============================================================
# 28. VISUALISATION 6
#     EMPLOYEES BY COUNTRY
# ============================================================

plt.figure(figsize=(8, 5))

clean["Country"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Employees by Country"
)

plt.xlabel(
    "Country"
)

plt.ylabel(
    "Number of Employees"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    output_folder / "06_country_counts.png",
    dpi=160
)

plt.show()


# ============================================================
# 29. VISUALISATION 7
#     AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

clean["Age"].astype(float).plot(
    kind="hist",
    bins=6
)

plt.title(
    "Age Distribution"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    output_folder / "07_age_distribution.png",
    dpi=160
)

plt.show()


# ============================================================
# 30. FINAL DATASET
# ============================================================

print("\n===================================")
print("FINAL CLEAN DATA")
print("===================================")

print(clean)