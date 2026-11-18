"""
MULTI-CLASS PATIENT TRIAGE CLASSIFICATION
Difficulty: Hard

Platform: Google Colab

Language: Python

Libraries: Pandas, NumPy

Submission: Google Colab Notebook


Problem Statement:
----------------
An emergency hospital system routes incoming patients to three care levels: General, Urgent, or ICU.
You must categorize medical attributes, configure a labeled dataset for multi-class supervised learning,
map supervised algorithms to specific tasks, and analyze class distributions.


Sample Input
----------------
Patient_ID,Heart_Rate_BPM,Systolic_BP,Oxygen_Sat_Pct,Age,Triage_Category
P_501,72,120,98,35,General
P_502,125,165,89,68,ICU
P_503,95,138,94,45,Urgent
P_504,68,115,99,28,General
P_505,140,180,85,75,ICU
P_506,88,130,95,52,Urgent
P_507,75,122,97,40,General
P_508,130,170,88,62,ICU
P_509,92,135,93,50,Urgent
P_510,70,118,98,30,General


TASKS TO IMPLEMENT
-------------------
TASK 1 — Dataset Construction: Load the 10 patient records into a Pandas DataFrame.  

TASK 2 — Supervised Paradigm Identification: Identify whether this is Classification or Regression. State whether the target is discrete or continuous.  

TASK 3 — Input-Output Splitting: Extract feature matrix $X$ and target vector $y$. Exclude non-predictive Patient_ID.  

TASK 4 — Multi-Class Target Distribution: Calculate absolute counts and percentages for General, Urgent, and ICU.  

TASK 5 — Group-Wise Feature Means: Compute mean Heart_Rate_BPM, Systolic_BP, and Oxygen_Sat_Pct for each triage level.  

TASK 6 — Supervised Algorithm Selection Guide: Map candidate algorithms (Logistic Regression, Decision Trees, K-Nearest Neighbors) to this task with justifications. 


EXPECTED OUTPUT
----------------
========== EMERGENCY PATIENT TRIAGE ANALYSIS ==========

Dataset Shape              : 10 Rows, 6 Columns
Supervised Paradigm        : Multi-Class Classification (Discrete Categorical Target)

Features (X)               : Heart_Rate_BPM, Systolic_BP, Oxygen_Sat_Pct, Age
Target (y)                 : Triage_Category

Class Distribution:
- General                  : 4 (40.0%)
- Urgent                   : 3 (30.0%)
- ICU                      : 3 (30.0%)

Triage Level Group Averages:
          Heart_Rate_BPM  Systolic_BP  Oxygen_Sat_Pct
General             71.25       118.25           98.00
Urgent              91.67       134.33           94.00
ICU                131.67       171.67           87.33

Algorithm Guidance: Decision Trees or Multi-Class Logistic Regression are recommended for interpretable clinical rule boundaries.
"""
print("========== EMERGENCY PATIENT TRIAGE ANALYSIS ==========")

import pandas as pd
import numpy as np

data = {
    "Patient_ID": [
        "P_501", "P_502", "P_503", "P_504", "P_505",
        "P_506", "P_507", "P_508", "P_509", "P_510"
    ],

    "Heart_Rate_BPM": [72, 125, 95, 68, 140, 88, 75, 130, 92, 70],

    "Systolic_BP": [
        120, 165, 138, 115, 180,
        130, 122, 170, 135, 118
    ],

    "Oxygen_Sat_Pct": [
        98, 89, 94, 99, 85,
        95, 97, 88, 93, 98
    ],

    "Age": [
        35, 68, 45, 28, 75,
        52, 40, 62, 50, 30
    ],

    "Triage_Category": [
        "General", "ICU", "Urgent", "General", "ICU",
        "Urgent", "General", "ICU", "Urgent", "General"
    ]
}

df = pd.DataFrame(data)

df

print("Dataset Shape    :",df.shape[0],"Rows,",df.shape[1],"Columns")

X = df[
    [
        "Heart_Rate_BPM",
        "Systolic_BP",
        "Oxygen_Sat_Pct",
        "Age"
    ]
]

y = df["Triage_Category"]

print("Feature (X):")
print(X)

print("\nTarget (y):")
print(y)

print("Features (X)               :", ", ".join(X.columns))
print("Target (y)                 :", y.name)

class_counts = y.value_counts()
print(class_counts)

class_percentages = y.value_counts(normalize=True)*100
print(class_percentages)

group_means = df.groupby("Triage_Category")[
    [
        "Heart_Rate_BPM",
        "Systolic_BP",
        "Oxygen_Sat_Pct"
    ]
].mean()

print(group_means)

print("\nAlgorithm Guidance:")
print("1. Logistic Regression  : Suitable for interpretable multi-class classification.")
print("2. Decision Trees       : Useful for interpretable threshold-based decision boundaries.")
print("3. K-Nearest Neighbors : Useful for predictions based on similarity between patients.")
print("\nDecision Trees or Multi-Class Logistic Regression can provide interpretable")
print("classification boundaries for this type of tabular problem.")