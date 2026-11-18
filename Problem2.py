"""E-COMMERCE CUSTOMER CHURN PREDICTION
Difficulty: Medium
Platform: Google Colab
Language: Python
Libraries: Pandas
Submission: Google Colab Notebook

Problem Statement:
---------------------
An e-commerce company wants to predict whether a customer will churn (cancel their account). Currently, customer service flags churn using manual rules (e.g., inactive for over 30 days and spent less than $50). The company wants to structure their raw customer logs into features and targets to move toward an ML-based prediction system. 

Sample Input
-------------
Customer_ID,Account_Age_Days,Monthly_Spend,Support_Calls,Inactivity_Days,Churn
C101,120,45.5,4,35,Yes
C102,450,120.0,0,5,No
C103,90,25.0,5,40,Yes
C104,300,85.0,1,12,No
C105,210,60.0,2,20,No
C106,60,15.0,6,45,Yes
C107,500,200.0,1,8,No

TASK 1 — Create the DatasetCreate the above dataset using Pandas. 
------------------------------------------------------------------ 
Expected Output:
Customer_ID  Account_Age_Days  Monthly_Spend  Support_Calls  Inactivity_Days Churn
0        C101               120           45.5              4               35   Yes
1        C102               450          120.0              0                5    No
2        C103                90           25.0              5               40   Yes
3        C104               300           85.0              1               12    No
4        C105               210           60.0              2               20    No
5        C106                60           15.0              6               45   Yes
6        C107               500          200.0              1                8    No


TASK 2 — Display Dataset Information
------------------------------------
Display the number of rows, columns, and column names.  

Expected Output:
----------------
Number of Rows    : 7
Number of Columns : 6

Columns:
Customer_ID
Account_Age_Days
Monthly_Spend
Support_Calls
Inactivity_Days
Churn

TASK 3 — Identify Features and Target
----------------------------------------
Identify input features vs. target label and explain why Customer_ID is excluded.  

Expected Output:

Features:
Account_Age_Days
Monthly_Spend
Support_Calls
Inactivity_Days

Target:
Churn

Customer_ID is an arbitrary unique identifier and contains no predictive pattern.

TASK 4 — Count Churned and Retained Customers
-----------------------------------------------------------------------------------------------
Calculate the count of churned vs. retained customers.  

Expected Output:
Churn Status Counts:
No     4
Yes    3


TASK 5 — Calculate Average Inactivity Days
-----------------------------------------
Calculate average inactivity days for churned vs. active customers. 
Expected Output:
Average Inactivity Days:
Churned Customers   : 40.0 days
Retained Customers : 11.25 days


TASK 6 — Apply a Rule-Based Prediction
-----------------------------
Apply company rule: 
IF Inactivity_Days > 30 AND Monthly_Spend < 50 THEN Churn = Yes ELSE No

Expected Output:
------------------
Customer_ID    Actual    Prediction
C101           Yes       Yes
C102           No        No
C103           Yes       Yes
C104           No        No
C105           No        No
C106           Yes       Yes
C107           No        No


TASK 7 — Calculate Rule Accuracy  
-------------------------------
Compare predictions against actual outcomes.

Expected Output:
-----------------
Correct Predictions : 7
Total Predictions   : 7
Rule-Based Accuracy : 100%


TASK 8 — Test New Customer Scenarios 
------------------------------------
Evaluate two new customers with the current rule: 

Customer A: Inactivity_Days = 32, Monthly_Spend = 150.0, Support_Calls = 6
Customer B: Inactivity_Days = 28, Monthly_Spend = 30.0, Support_Calls = 8

Expected Output:
------------------
Customer A Prediction : No
Customer B Prediction : No

Analysis: Customer B has 8 support calls and high inactivity, but the rule outputs "No" because Monthly_Spend isn't under $50. Rule-based systems miss complex combined factors.


TASK 9 — Identify Features & Labels in Real-World Context
-----------------------------
Display the foundational definitions of Data, Features, and Target Labels. 

Expected Output:
------------------
FEATURES & LABELS FOUNDATIONS

Features (X) : Input measures provided to the model (e.g., Inactivity_Days, Support_Calls).
Target (y)   : The ground-truth answer the model tries to learn/predict (e.g., Churn).

FINAL EXPECTED OUTPUT
---------------------
========== CUSTOMER CHURN ANALYSIS ==========

Total Customers        : 7
Churned Customers      : 3
Retained Customers     : 4

Average Inactivity
Churned                : 40.0 days
Retained               : 11.25 days

Rule-Based Accuracy    : 100%

New Customer A Pred    : No
New Customer B Pred    : No

Conclusion:
Hardcoded rules fail when missing complex interactions (like high support calls).
ML models dynamically balance all input features simultaneously. """

# task 1
import pandas as pd

data = {
    "Customer_ID": ["C101", "C102", "C103", "C104", "C105", "C106", "C107"],
    "Account_Age_Days": [120, 450, 90, 300, 210, 60, 500],
    "Monthly_Spend": [45.5, 120.0, 25.0, 85.0, 60.0, 15.0, 200.0],
    "Support_Calls": [4, 0, 5, 1, 2, 6, 1],
    "Inactivity_Days": [35, 5, 40, 12, 20, 45, 8],
    "Churn": ["Yes", "No", "Yes", "No", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

print(df)

# task2
print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])

print("\nColumns:")

for column in df.columns:
    print(column)
    
# task3
features = [
    "Account_Age_Days",
    "Monthly_Spend",
    "Support_Calls",
    "Inactivity_Days"
]

target = "Churn"

print("Features:")
for feature in features:
    print(feature)

print("\nTarget:")
print(target)

print("\nCustomer_ID is an arbitrary unique identifier and contains no predictive pattern.")

# task4
churn_counts = df["Churn"].value_counts()

print("Churn Status Counts:")
print(churn_counts)

# task5
average_inactivity = df.groupby("Churn")["Inactivity_Days"].mean()

print("Average Inactivity")

print("Churned Customers   :", average_inactivity["Yes"], "days")
print("Retained Customers :", average_inactivity["No"], "days")

# task6
df["Prediction"] = (
    (df["Inactivity_Days"] > 30) &
    (df["Monthly_Spend"] < 50)
).map({True: "Yes", False: "No"})

result = df[["Customer_ID", "Churn", "Prediction"]]

result = result.rename(columns={
    "Churn": "Actual"
})

print(result.to_string(index=False))

# task7
correct_predictions = (df["Churn"] == df["Prediction"]).sum()
total_predictions = len(df)

accuracy = (correct_predictions / total_predictions) * 100

print("Correct Predictions :", correct_predictions)
print("Total Predictions   :", total_predictions)
print("Rule-Based Accuracy :",f"{accuracy:.0f}%")

# task8
new_customers = pd.DataFrame({
    "Customer_ID": ["A", "B"],
    "Inactivity_Days": [32, 28],
    "Monthly_Spend": [150.0, 30.0],
    "Support_Calls": [6, 8]
})

new_customers["Prediction"] = (
    (new_customers["Inactivity_Days"] > 30) &
    (new_customers["Monthly_Spend"] < 50)
).map({True: "Yes", False: "No"})
print("Customer A Prediction :", new_customers.loc[0, "Prediction"])
print("Customer B Prediction :", new_customers.loc[1, "Prediction"])

# task9
print("FEATURES & LABELS FOUNDATIONS")
print()
print("Features (X) : Input measures provided to the model (e.g., Inactivity_Days, Support_Calls).")
print("Target (y)   : The ground-truth answer the model tries to learn/predict (e.g., Churn).")

print("\n========== CUSTOMER CHURN ANALYSIS ==========")

print("Total Customers        :", len(df))

churned_customers = (df["Churn"] == "Yes").sum()
retained_customers = (df["Churn"] == "No").sum()

print("Churned Customers      :", churned_customers)
print("Retained Customers     :", retained_customers)

print("\nAverage Inactivity")

print("Churned                :", average_inactivity["Yes"], "days")
print("Retained               :", average_inactivity["No"], "days")

print("\nRule-Based Accuracy    :", f"{accuracy:.0f}%")

print("New Customer A Pred    :", new_customers.loc[0, "Prediction"])
print("New Customer B Pred    :", new_customers.loc[1, "Prediction"])

print("\nConclusion:")
print("Hardcoded rules can fail when they do not consider complex interactions (like high support calls).")
print("ML models dynamically balance all input features simultaneously.")