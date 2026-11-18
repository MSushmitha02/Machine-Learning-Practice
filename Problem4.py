"""FINANCIAL CREDIT CARD FRAUD & RISK AUDITING
------------------------------------------------------
Platform: Google Colab

Language: Python

Libraries: Pandas, NumPy

Submission: Google Colab Notebook

Problem Statement:
-------------------
A banking institution processes thousands of daily transactions. Risk analysts currently rely on static security thresholds to flag fraudulent transactions. You must join 4 separate relational tables (Transactions, Card Accounts, Terminal Devices, and Risk Blacklists) covering 50 records to construct a dataset, measure baseline rule performance, and evaluate ML feature/label representations.


Input Files & Data Setup
--------------------------
import pandas as pd
import numpy as np

# File 1: Transactions (50 Records)
tx_data = {
    'Tx_ID': [f'TX_{1000+i}' for i in range(50)],
    'Account_ID': [f'ACC_{(i%10)+101}' for i in range(50)],
    'Terminal_ID': [f'TRM_{(i%8)+501}' for i in range(50)],
    'Tx_Amount': [15.2, 1250.0, 45.0, 3200.0, 8.5, 980.0, 2100.0, 12.0, 4500.0, 65.0, 1100.0, 25.0, 2800.0, 80.0, 1750.0, 5.0, 3900.0, 18.0, 1300.0, 95.0, 2400.0, 30.0, 4100.0, 40.0, 1600.0, 15.0, 3100.0, 55.0, 2200.0, 10.0, 1450.0, 70.0, 4800.0, 22.0, 1900.0, 12.0, 2600.0, 85.0, 3300.0, 35.0, 1200.0, 60.0, 4200.0, 18.0, 1800.0, 45.0, 2900.0, 25.0, 2100.0, 90.0],
    'Tx_Hour': [14, 2, 11, 3, 16, 1, 4, 18, 23, 10, 2, 15, 1, 12, 3, 19, 4, 13, 2, 9, 3, 17, 1, 11, 4, 20, 2, 8, 3, 15, 1, 14, 4, 18, 2, 12, 3, 10, 1, 16, 2, 11, 4, 21, 3, 9, 2, 13, 1, 7],
    'Is_Fraud': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# File 2: Accounts Metadata (10 Records)
acc_data = {
    'Account_ID': [f'ACC_{i}' for i in range(101, 111)],
    'Avg_Monthly_Tx_Vol': [1200.0, 300.0, 1500.0, 400.0, 2000.0, 250.0, 500.0, 1800.0, 350.0, 1100.0],
    'Account_Age_Months': [48, 6, 60, 3, 84, 2, 12, 72, 4, 36]
}

# File 3: Terminal Devices (8 Records)
trm_data = {
    'Terminal_ID': [f'TRM_{i}' for i in range(501, 509)],
    'Terminal_Risk_Score': [0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.85, 0.05],
    'Is_Foreign_Location': [0, 1, 0, 1, 0, 1, 1, 0]
}

# File 4: Risk Blacklists (50 Records)
blk_data = {
    'Tx_ID': [f'TX_{1000+i}' for i in range(50)],
    'High_Risk_IP_Flag': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df_tx = pd.DataFrame(tx_data)
df_acc = pd.DataFrame(acc_data)
df_trm = pd.DataFrame(trm_data)
df_blk = pd.DataFrame(blk_data)


TASKS TO IMPLEMENT
--------------------
TASK 1 — Enterprise 4-Table Relational Merge: Perform sequential inner/left joins on df_tx, df_acc, df_trm, and df_blk to build a single master fraud analytical table. 

TASK 2 — Schema Validation: Output dataset dimensions, inspect column data types, and check missing value distributions. 

TASK 3 — Feature Space Definition: Classify inputs into predictive features versus operational identifiers (Tx_ID, Account_ID, Terminal_ID). Explicitly specify binary target Is_Fraud.  

TASK 4 — Fraud Class Aggregation: Compute class balances (0 vs. 1) and calculate median Tx_Amount and Terminal_Risk_Score per class.  

TASK 5 — Rule-Based Fraud System Simulation: Apply security rule: IF (Tx_Amount > 1000 AND Tx_Hour IN [1,2,3,4]) OR High_Risk_IP_Flag == 1 THEN Prediction = 1 ELSE 0.  

TASK 6 — Quantitative Fraud Detection Performance: Compute accuracy, correct detections, false alarms, and missed fraud occurrences.  

TASK 7 — Train vs Predict Lifecycle Blueprint: Format 5 records representing live transaction stream (excluding target Is_Fraud). Explain how training phase differs from live prediction phase. 

EXPECTED FINAL OUTPUT
----------------------
========== FINANCIAL CREDIT CARD FRAUD AUDIT ==========

Master Dataset Records     : 50
Total Feature Attributes   : 9
Class Distribution (Fraud) : Legitimate = 26 (52.0%), Fraudulent = 24 (48.0%)

Group Medians:
- Median Tx Amount         : Fraud = $2,250.00 | Legitimate = $32.50
- Terminal Risk Score      : Fraud = 0.80      | Legitimate = 0.10

Static Security Rule Evaluation:
- Rule Accuracy            : 100.00%
- False Alarms (FP)        : 0
- Missed Frauds (FN)       : 0

Production Pipeline Note:
During training, both feature matrix X and ground-truth y are passed to fit parameters. I
n live production inference, only transaction streams (X) are provided to output risk scores. """

import numpy as np
import pandas as pd

tx_data = {
    'Tx_ID': [f'TX_{1000+i}' for i in range(50)],
    'Account_ID': [f'ACC_{(i%10)+101}' for i in range(50)],
    'Terminal_ID': [f'TRM_{(i%8)+501}' for i in range(50)],
    'Tx_Amount': [15.2, 1250.0, 45.0, 3200.0, 8.5, 980.0, 2100.0, 12.0, 4500.0, 65.0, 1100.0, 25.0, 2800.0, 80.0, 1750.0, 5.0, 3900.0, 18.0, 1300.0, 95.0, 2400.0, 30.0, 4100.0, 40.0, 1600.0, 15.0, 3100.0, 55.0, 2200.0, 10.0, 1450.0, 70.0, 4800.0, 22.0, 1900.0, 12.0, 2600.0, 85.0, 3300.0, 35.0, 1200.0, 60.0, 4200.0, 18.0, 1800.0, 45.0, 2900.0, 25.0, 2100.0, 90.0],
    'Tx_Hour': [14, 2, 11, 3, 16, 1, 4, 18, 23, 10, 2, 15, 1, 12, 3, 19, 4, 13, 2, 9, 3, 17, 1, 11, 4, 20, 2, 8, 3, 15, 1, 14, 4, 18, 2, 12, 3, 10, 1, 16, 2, 11, 4, 21, 3, 9, 2, 13, 1, 7],
    'Is_Fraud': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

acc_data = {
    'Account_ID': [f'ACC_{i}' for i in range(101, 111)],
    'Avg_Monthly_Tx_Vol': [1200.0, 300.0, 1500.0, 400.0, 2000.0, 250.0, 500.0, 1800.0, 350.0, 1100.0],
    'Account_Age_Months': [48, 6, 60, 3, 84, 2, 12, 72, 4, 36]
}

trm_data = {
    'Terminal_ID': [f'TRM_{i}' for i in range(501, 509)],
    'Terminal_Risk_Score': [0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.85, 0.05],
    'Is_Foreign_Location': [0, 1, 0, 1, 0, 1, 1, 0]
}

blk_data = {
    'Tx_ID': [f'TX_{1000+i}' for i in range(50)],
    'High_Risk_IP_Flag': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df_tx = pd.DataFrame(tx_data)
df_acc = pd.DataFrame(acc_data)
df_trm = pd.DataFrame(trm_data)
df_blk = pd.DataFrame(blk_data)

df_master = df_tx.merge(df_acc,on='Account_ID',how='left')
df_master = df_master.merge(df_trm,on='Terminal_ID',how='left')
df_master = df_master.merge(df_blk,on='Tx_ID',how='left')

print(df_master.head())
print("\nMaster Shape:", df_master.shape)

print("Number of Rows    :", df_master.shape[0])
print("Number of Columns :", df_master.shape[1])

print("\nColumns:")

for column in df_master.columns:
    print(column)
    
print("\nData Types:")
print(df_master.dtypes)

missing_values = df_master.isnull().sum()

print("\nMissing Values:")
print(missing_values)

target = 'Is_Fraud'

identifier_columns = [
    'Tx_ID',
    'Account_ID',
    'Terminal_ID'
]

feature_columns = [
    column
    for column in df_master.columns
    if column not in identifier_columns + [target]
]

X = df_master[feature_columns]
y = df_master[target]

print("Predictive Features:")

for feature in feature_columns:
    print("-", feature)

print("\nTarget:")
print("-", target)

print("\nOperational Identifiers:")

for column in identifier_columns:
    print("-", column)
    
class_counts = df_master['Is_Fraud'].value_counts()

print("Fraud Class Distribution:")
print(class_counts)

class_percentages = (
    df_master['Is_Fraud']
    .value_counts(normalize=True)
    .mul(100)
)

print("\nClass Percentages:")

for class_value in [0, 1]:
    print(
        class_value,
        "=",
        f"{class_percentages[class_value]:.1f}%"
    )
    
median_metrics = df_master.groupby('Is_Fraud')[
    ['Tx_Amount', 'Terminal_Risk_Score']
].median()

print(median_metrics)

rule_condition = (
    ((df_master['Tx_Amount'] > 1000) & (df_master['Tx_Hour'].isin([1, 2, 3, 4])))|
    (df_master['High_Risk_IP_Flag'] == 1)
)

df_master['Prediction'] = np.where(rule_condition,1,0)

print(
    df_master[
        [
            'Tx_ID',
            'Tx_Amount',
            'Tx_Hour',
            'High_Risk_IP_Flag',
            'Is_Fraud',
            'Prediction'
        ]
    ].to_string(index=False)
)

correct_predictions = (
    df_master['Is_Fraud'] ==
    df_master['Prediction']
).sum()

total_predictions = len(df_master)

accuracy = (
    correct_predictions /
    total_predictions
) * 100

false_positives = (
    (df_master['Is_Fraud'] == 0) &
    (df_master['Prediction'] == 1)).sum()

false_negatives = (
    (df_master['Is_Fraud'] == 1) &
    (df_master['Prediction'] == 0)).sum()

print("Static Rule Accuracy       :", f"{accuracy:.2f}%")
print("False Alarms               :", false_positives)
print("Missed Frauds              :", false_negatives)

confusion_matrix = pd.crosstab(
    df_master['Is_Fraud'],
    df_master['Prediction'],
    rownames=['Actual'],
    colnames=['Prediction']
)

print(confusion_matrix)

live_transactions = df_master[
    [
        'Tx_ID',
        'Account_ID',
        'Terminal_ID',
        'Tx_Amount',
        'Tx_Hour',
        'Avg_Monthly_Tx_Vol',
        'Account_Age_Months',
        'Terminal_Risk_Score',
        'Is_Foreign_Location',
        'High_Risk_IP_Flag'
    ]
].head(5).copy()

print("LIVE TRANSACTION STREAM")
print(live_transactions.to_string(index=False))

X_train = df_master[feature_columns]
y_train = df_master['Is_Fraud']

X_live = live_transactions[feature_columns]

print("========== FINANCIAL CREDIT CARD FRAUD AUDIT ==========")

print(
    "Master Dataset Records     :",
    len(df_master)
)

print(
    "Total Feature Attributes   :",
    len(feature_columns)
)

print(
    f"\nClass Distribution (Fraud) : Legitimate = {class_counts[0]} ({class_percentages[0]:.1f}%), Fraudulent = {class_counts[1]} ({class_percentages[1]:.1f}%)")

print("\nGroup Medians:")

print(
    f"- Median Tx Amount         : "
    f"Fraud = ${median_metrics.loc[1, 'Tx_Amount']:,.2f}"
    f" | "
    f"Legitimate = ${median_metrics.loc[0, 'Tx_Amount']:,.2f}"
)

print(
    f"- Terminal Risk Score      : "
    f"Fraud = {median_metrics.loc[1, 'Terminal_Risk_Score']:.2f}"
    f"      | "
    f"Legitimate = {median_metrics.loc[0, 'Terminal_Risk_Score']:.2f}"
)

print("\nStatic Security Rule Evaluation:")
print(
    f"- Rule Accuracy            : {accuracy:.2f}%"
)

print(
    f"- False Alarms (FP)        : {false_positives}"
)

print(
    f"- Missed Frauds (FN)       : {false_negatives}"
)

print("\nProduction Pipeline Note:")

print(
    "During training, feature matrix X and ground-truth y are passed to fit parameters. In live production inference, only transaction streams (X) are provided to output risk scores."
)