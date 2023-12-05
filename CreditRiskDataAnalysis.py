import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load data and display the first few rows
credit_risk = pd.read_csv("UCI_credit_card.csv")
print(credit_risk.head())

# Drop unnecessary column 'ID'
df = credit_risk.drop(columns=["ID"])

# Display data statistics and check for missing values
print(df.describe())
print(df.isnull().sum())

# Standardize 'EDUCATION' and 'MARRIAGE' columns
df['EDUCATION'].replace({0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 1}, inplace=True)
df['MARRIAGE'].replace({0: 1, 1: 1, 2: 2, 3: 3}, inplace=True)

# Visualize target distribution
plt.figure(figsize=(6, 6))
ax = sns.countplot(df['default.payment.next.month'])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0, 1], labels=["Not Defaulted", "Defaulted"])
plt.title("Target Distribution")
plt.show()

# Visualize age distribution
plt.figure(figsize=(6, 6))
sns.distplot(df['AGE'], kde=True)
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.title("Age Distribution")
plt.show()

# Visualize gender distribution
plt.figure(figsize=(6, 6))
ax = sns.countplot('SEX', hue='default.payment.next.month', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0, 1], labels=["Male", "Female"])
plt.title("Gender Distribution")
plt.show()

# Visualize education distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot('EDUCATION', hue='default.payment.next.month', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0, 1, 2, 3], labels=["Graduate School", "University", 'High School', 'Others'])
plt.title("Education Distribution")
plt.show()

# Visualize marriage distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot('MARRIAGE', hue='default.payment.next.month', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0, 1, 2], labels=["Married", "Single", 'Others'])
plt.title("Marriage Distribution")
plt.show()

# Visualize payment structure vs bill amount
plt.subplots(figsize=(20, 10))
for i in range(1, 6):
    plt.subplot(2, 3, i)
    plt.scatter(x=df[f'PAY_AMT{i}'], y=df[f'BILL_AMT{i+1}'], c=plt.cm.rainbow(i / 6), s=1)
    plt.xlabel(f'PAY_AMT{i}')
    plt.ylabel(f'BILL_AMT{i+1}')
plt.suptitle('Payment Structure vs Bill Amount in the Last 6 Months', fontsize=15)
plt.show()

# Prepare data for modeling
X = df.drop(['default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

# Standardize features and split the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply SMOTE for oversampling
print("Before oversampling:", Counter(y_train))
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After oversampling:", Counter(y_train))
