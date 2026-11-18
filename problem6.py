""" "REAL ESTATE ASSET VALUATION (REGRESSION)

Platform: Google Colab

Language: Python

Libraries: Pandas, NumPy

Submission: Google Colab Notebook

Problem Statement
A commercial property investment firm evaluates valuation models for real estate acquisitions. Rather than relying on a single dataset, you are provided with 3 relational tables: Property Listings, Neighborhood Analytics, and Macroeconomic Indicators. You must perform a multi-table relational join across 50 records, analyze continuous variable correlations, build feature-target mappings for regression, and formulate explicit task outputs.

Input Files & Data Setup
--------------------------
import pandas as pd
import numpy as np

# File 1: Property Listings (50 Records)
listings_data = {
    'Property_ID': [f'PROP_{100+i}' for i in range(50)],
    'Neighborhood_ID': [f'NGH_{(i%5)+1}' for i in range(50)],
    'Area_SqFt': [2500, 1200, 4000, 1800, 3200, 950, 2800, 1500, 4500, 2100, 3100, 1300, 4200, 1900, 3300, 1000, 4600, 1600, 3400, 2200, 2900, 1400, 4100, 2000, 3500, 1100, 4700, 1700, 3600, 2300, 3000, 1250, 4300, 1850, 3250, 1050, 4400, 1650, 3350, 2150, 2850, 1350, 3950, 1750, 3450, 1150, 4800, 1550, 3700, 2250],
    'Bedrooms': [4, 2, 5, 3, 4, 1, 4, 2, 5, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3, 4, 2, 5, 3, 4, 1, 5, 2, 4, 3],
    'Building_Age_Yrs': [8, 25, 2, 15, 5, 30, 10, 18, 1, 12, 6, 22, 3, 14, 4, 28, 1, 16, 5, 11, 9, 20, 2, 13, 4, 26, 1, 17, 3, 10, 7, 24, 2, 15, 5, 29, 1, 18, 4, 12, 8, 21, 3, 16, 5, 27, 1, 19, 3, 11],
    'Price_USD_M': [1.25, 0.42, 2.85, 0.68, 1.95, 0.29, 1.40, 0.55, 3.20, 0.92, 1.80, 0.48, 2.95, 0.72, 1.90, 0.32, 3.35, 0.58, 2.05, 0.98, 1.48, 0.51, 2.88, 0.78, 2.15, 0.35, 3.45, 0.62, 2.25, 1.05, 1.65, 0.45, 3.05, 0.70, 1.98, 0.31, 3.15, 0.60, 2.10, 0.95, 1.42, 0.49, 2.75, 0.65, 2.20, 0.38, 3.50, 0.56, 2.30, 1.02]
}

# File 2: Neighborhood Analytics (5 Records)
neighborhood_data = {
    'Neighborhood_ID': [f'NGH_{i}' for i in range(1, 6)],
    'Distance_City_KM': [2.5, 18.0, 5.0, 12.0, 1.2],
    'School_Rating': [9.2, 5.5, 8.8, 6.8, 9.6],
    'Crime_Index': [12.0, 45.0, 18.0, 32.0, 8.5]
}

# File 3: Macroeconomic Indicators (50 Records)
macro_data = {
    'Property_ID': [f'PROP_{100+i}' for i in range(50)],
    'Interest_Rate_Pct': [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4, 6.4],
    'Property_Tax_Rate': [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013]
}

df_listings = pd.DataFrame(listings_data)
df_neighborhood = pd.DataFrame(neighborhood_data)
df_macro = pd.DataFrame(macro_data)


TASKS TO IMPLEMENT
-------------------
TASK 1 — Multi-Source Relational Data Merging
Merge df_listings, df_neighborhood, and df_macro using primary/foreign key joins (Property_ID, Neighborhood_ID).

Expected Output:
  Property_ID Neighborhood_ID  Area_SqFt  ...  School_Rating  Crime_Index  Interest_Rate_Pct
0    PROP_100          NGH_1       2500  ...            9.2         12.0                6.5
1    PROP_101          NGH_2       1200  ...            5.5         45.0                6.5
2    PROP_102          NGH_3       4000  ...            8.8         18.0                6.5
3    PROP_103          NGH_4       1800  ...            6.8         32.0                6.5
4    PROP_104          NGH_5       3200  ...            9.6          8.5                6.5


TASK 2 — Supervised Paradigm Verification
Verify dataset shape, column names, and confirm target type (Continuous Numerical vs. Discrete).

Expected Output:
Total Records           : 50
Total Combined Columns : 11
Target Variable        : Price_USD_M
Target Data Type       : float64
Supervised Paradigm    : Regression (Continuous Target)


TASK 3 — Feature Space & Target Structuring
Extract Feature Matrix (X) and Target Vector (y). Exclude non-predictive identifiers (Property_ID, Neighborhood_ID).

Expected Output:
Features (X) Columns:
- Area_SqFt
- Bedrooms
- Building_Age_Yrs
- Distance_City_KM
- School_Rating
- Crime_Index
- Interest_Rate_Pct
- Property_Tax_Rate

Target (y):
- Price_USD_M


TASK 4 — Pearson Correlation Matrix Analysis
Compute Pearson correlation coefficients of all numeric features relative to Price_USD_M.

Expected Output:
Pearson Correlation with Price_USD_M:
Area_SqFt           : +0.983
Bedrooms            : +0.925
School_Rating       : +0.864
Building_Age_Yrs    : -0.887
Crime_Index         : -0.892
Distance_City_KM    : -0.915
Interest_Rate_Pct   : -0.142
Property_Tax_Rate   : -0.118

Strongest Positive Feature : Area_SqFt (+0.983)
Strongest Negative Feature : Distance_City_KM (-0.915)


TASK 5 — Sub-Group Property Valuation Metrics
Group dataset into High Value (> $1.5M) and Moderate Value (<= $1.5M) properties. Compute mean Area_SqFt, Distance_City_KM, and School_Rating.

Expected Output:
High Value Properties (> $1.5M):
- Total Properties   : 23
- Mean Area          : 3,760.87 SqFt
- Mean Distance      : 2.87 KM
- Mean School Rating : 9.20

Moderate Value Properties (<= $1.5M):
- Total Properties   : 27
- Mean Area          : 1,770.37 SqFt
- Mean Distance      : 11.22 KM
- Mean School Rating : 6.85


TASK 6 — Algorithm Selection Guide
Provide algorithm mapping for continuous regression modeling with trade-offs.

Expected Output:
REGRESSION ALGORITHM MAP:
1. Linear Regression    : Baseline linear relationship mapping; fast and interpretable.
2. Ridge / Lasso        : Handles multicollinearity between Area_SqFt and Bedrooms.
3. Decision Tree Regressor : Captures non-linear local neighborhood threshold effects.



FINAL EXPECTED OUTPUT
---------------------
========== COMMERCIAL PROPERTY REGRESSION ANALYSIS ==========

Master Dataset Records     : 50
Total Feature Attributes   : 8
Supervised Paradigm        : Regression (Continuous Target)

Target Metrics (Price_USD_M):
- Minimum Price            : $0.29M
- Maximum Price            : $3.50M
- Mean Price               : $1.49M

Key Feature Correlations:
- Top Positive Predictor   : Area_SqFt (+0.983)
- Top Negative Predictor   : Distance_City_KM (-0.915)

Valuation Sub-Group Profiling:
- High Value (> $1.5M)     : Mean Area = 3,760.87 SqFt | Mean Distance = 2.87 KM | Mean School Rating = 9.20
- Moderate Value (<= $1.5M): Mean Area = 1,770.37 SqFt | Mean Distance = 11.22 KM | Mean School Rating = 6.85

Conclusion:
Multi-table integration confirms property prices are strongly driven by continuous physical attributes (Area) and spatial neighborhood metrics (Distance/School Rating). Linear and Regularized Regression models are optimal candidate baselines."""

import pandas as pd
import numpy as np

listings_data = {

    'Property_ID': [f'PROP_{100+i}' for i in range(50)],

    'Neighborhood_ID': [f'NGH_{(i%5)+1}' for i in range(50)],

    'Area_SqFt': [
        2500, 1200, 4000, 1800, 3200, 950, 2800, 1500, 4500, 2100,
        3100, 1300, 4200, 1900, 3300, 1000, 4600, 1600, 3400, 2200,
        2900, 1400, 4100, 2000, 3500, 1100, 4700, 1700, 3600, 2300,
        3000, 1250, 4300, 1850, 3250, 1050, 4400, 1650, 3350, 2150,
        2850, 1350, 3950, 1750, 3450, 1150, 4800, 1550, 3700, 2250
    ],

    'Bedrooms': [
        4, 2, 5, 3, 4, 1, 4, 2, 5, 3,
        4, 2, 5, 3, 4, 1, 5, 2, 4, 3,
        4, 2, 5, 3, 4, 1, 5, 2, 4, 3,
        4, 2, 5, 3, 4, 1, 5, 2, 4, 3,
        4, 2, 5, 3, 4, 1, 5, 2, 4, 3
    ],
    'Building_Age_Yrs': [
        8, 25, 2, 15, 5, 30, 10, 18, 1, 12,
        6, 22, 3, 14, 4, 28, 1, 16, 5, 11,
        9, 20, 2, 13, 4, 26, 1, 17, 3, 10,
        7, 24, 2, 15, 5, 29, 1, 18, 4, 12,
        8, 21, 3, 16, 5, 27, 1, 19, 3, 11
    ],

    'Price_USD_M': [
        1.25, 0.42, 2.85, 0.68, 1.95, 0.29, 1.40, 0.55, 3.20, 0.92,
        1.80, 0.48, 2.95, 0.72, 1.90, 0.32, 3.35, 0.58, 2.05, 0.98,
        1.48, 0.51, 2.88, 0.78, 2.15, 0.35, 3.45, 0.62, 2.25, 1.05,
        1.65, 0.45, 3.05, 0.70, 1.98, 0.31, 3.15, 0.60, 2.10, 0.95,
        1.42, 0.49, 2.75, 0.65, 2.20, 0.38, 3.50, 0.56, 2.30, 1.02
    ]
}

neighborhood_data = {

    'Neighborhood_ID': [
        f'NGH_{i}' for i in range(1, 6)
    ],

    'Distance_City_KM': [
        2.5, 18.0, 5.0, 12.0, 1.2
    ],

    'School_Rating': [
        9.2, 5.5, 8.8, 6.8, 9.6
    ],

    'Crime_Index': [
        12.0, 45.0, 18.0, 32.0, 8.5
    ]
}

macro_data = {

    'Property_ID': [
        f'PROP_{100+i}' for i in range(50)
    ],

    'Interest_Rate_Pct': (
        [6.5] * 10 +
        [6.8] * 10 +
        [7.0] * 10 +
        [6.2] * 10 +
        [6.4] * 10
    ),

    'Property_Tax_Rate': (
        [0.012] * 10 +
        [0.014] * 10 +
        [0.015] * 10 +
        [0.011] * 10 +
        [0.013] * 10
    )
}

df_listings = pd.DataFrame(listings_data)
df_neighborhood = pd.DataFrame(neighborhood_data)
df_macro = pd.DataFrame(macro_data)

master_df = (
    df_listings.merge(df_neighborhood,on='Neighborhood_ID',how='inner')
    .merge(df_macro,on='Property_ID',how='inner')
)

target = master_df['Price_USD_M']

feature_columns = [
    'Area_SqFt',
    'Bedrooms',
    'Building_Age_Yrs',
    'Distance_City_KM',
    'School_Rating',
    'Crime_Index',
    'Interest_Rate_Pct',
    'Property_Tax_Rate'
]

X = master_df[feature_columns]

y = master_df['Price_USD_M']

correlations = (
    master_df
    .select_dtypes(include=np.number)
    .corr()['Price_USD_M']
    .drop('Price_USD_M')
)

strongest_positive_feature = correlations.idxmax()
strongest_positive_value = correlations.max()

strongest_negative_feature = correlations.idxmin()
strongest_negative_value = correlations.min()

high_value = master_df[
    master_df['Price_USD_M']>1.5
]

moderate_value = master_df[
    master_df["Price_USD_M"]<=1.5
]

high_area = high_value['Area_SqFt'].mean()
high_distance = high_value['Distance_City_KM'].mean()
high_school = high_value['School_Rating'].mean()

moderate_area = moderate_value['Area_SqFt'].mean()
moderate_distance = moderate_value['Distance_City_KM'].mean()
moderate_school = moderate_value['School_Rating'].mean()

print("========== COMMERCIAL PROPERTY REGRESSION ANALYSIS ==========")


# Dataset information

print("\nMaster Dataset Records      :", master_df.shape[0])
print("Total Feature Attributes   :", len(feature_columns))

# Regression information

print("\nSupervised Paradigm        : Regression (Continuous Target)")

print("Target Variable            :", y.name)

print("Target Data Type           :", y.dtype)

print("\nFeatures (X) Columns:")

for column in X.columns:
    print("-", column)


print("\nTarget (y):")

print("-", y.name)

print("\nTarget Metrics (Price_USD_M):")

print(
    f"- Minimum Price            : ${y.min():.2f}M"
)

print(
    f"- Maximum Price            : ${y.max():.2f}M"
)

print(
    f"- Mean Price               : ${y.mean():.2f}M"
)

print("\nPearson Correlation with Price_USD_M:")

for feature in [
    'Area_SqFt',
    'Bedrooms',
    'School_Rating',
    'Building_Age_Yrs',
    'Crime_Index',
    'Distance_City_KM',
    'Interest_Rate_Pct',
    'Property_Tax_Rate'
]:

    print(
        f"{feature:<22}: {correlations[feature]:+.3f}"
    )


print(
    f"\nStrongest Positive Feature : "
    f"{strongest_positive_feature} "
    f"({strongest_positive_value:+.3f})"
)

print(
    f"Strongest Negative Feature : "
    f"{strongest_negative_feature} "
    f"({strongest_negative_value:+.3f})"
)

print("\nValuation Sub-Group Profiling:")

print("\nHigh Value Properties (> $1.5M):")

print(
    f"- Total Properties         : {len(high_value)}"
)

print(
    f"- Mean Area                : "
    f"{high_area:.2f} SqFt"
)

print(
    f"- Mean Distance            : "
    f"{high_distance:.2f} KM"
)

print(
    f"- Mean School Rating      : "
    f"{high_school:.2f}"
)

print("\nModerate Value Properties (<= $1.5M):")

print(
    f"- Total Properties         : {len(moderate_value)}"
)

print(
    f"- Mean Area                : "
    f"{moderate_area:.2f} SqFt"
)

print(
    f"- Mean Distance            : "
    f"{moderate_distance:.2f} KM"
)

print(
    f"- Mean School Rating      : "
    f"{moderate_school:.2f}"
)

print("\n" + "=" * 65)
print("REGRESSION ALGORITHM MAP:")
print("=" * 65)

print(
    "\n1. Linear Regression"
    " : Baseline linear relationship mapping; "
    "fast and interpretable."
)

print(
    "\n2. Ridge / Lasso"
    " : Handles multicollinearity and "
    "can regularize feature coefficients."
)

print(
    "\n3. Decision Tree Regressor"
    " : Captures non-linear relationships "
    "and local threshold effects."
)

print("\nConclusion:")

print(
    "Multi-table integration confirms property prices are strongly driven by continuous physical attributes (Area) and spatial neighborhood metrics (Distance/School Rating). Linear and Regularized Regression models are optimal candidate baselines. "
)