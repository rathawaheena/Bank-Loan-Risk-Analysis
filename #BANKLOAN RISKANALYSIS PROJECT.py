import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("loan-prediction-dataset.csv")

print("First 5 rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

print("\nAfter Cleaning Missing Values")
print(df.isnull().sum())


print("\nLoan Status Count")
print(df['Loan_Status'].value_counts())


sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()


sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Credit History vs Loan Status")
plt.show()


plt.scatter(df['ApplicantIncome'], df['LoanAmount'])
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.title("Income vs Loan Amount")
plt.show()


df['Risk_Level'] = np.where(df['Credit_History'] == 0, "High Risk", "Low Risk")

print("\nRisk Level Sample")
print(df[['ApplicantIncome','LoanAmount','Credit_History','Risk_Level']].head())


risk_counts = df['Risk_Level'].value_counts()

print("\nRisk Distribution")
print(risk_counts)

risk_counts.plot(kind='bar')
plt.title("High Risk vs Low Risk Customers")
plt.show()


df.to_csv("processed_loan_data.csv", index=False)

print("\nProject Completed Successfully")