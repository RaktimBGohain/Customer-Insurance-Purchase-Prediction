import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = r"C:\Users\This PC\Downloads\IntrnForte\Project 1\insurance_data.csv"
df = pd.read_csv(file_path)

print(" Dataset Loaded Successfully!")
print(df.head())

plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=20)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Salary'], kde=True, bins=20)
plt.title('Distribution of Salary')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Insurance_Purchase', data=df)
plt.title('Insurance Purchase Breakdown')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Salary', hue='Insurance_Purchase', data=df, palette='RdYlGn', s=100)
plt.title('Age vs Salary: Who Buys Insurance?')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend(title='Insurance Purchase', loc='upper left')
plt.show()

X = df[['Age', 'Salary']]
y = df['Insurance_Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression()
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)

log_reg_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

y_pred_log_reg = log_reg_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n {model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

evaluate_model(y_test, y_pred_log_reg, 'Logistic Regression')
evaluate_model(y_test, y_pred_dt, 'Decision Tree')
evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_svm, 'Support Vector Machine')

test_data = np.array([[30, 87000], [40, 0], [40, 100000], [50, 0]])

best_model = rf_model
predictions = best_model.predict(test_data)

for i, prediction in enumerate(predictions):
    print(f"Prediction for Age {test_data[i][0]}, Salary {test_data[i][1]}: {'Yes' if prediction == 1 else 'No'}")

print("\n Model Training and Evaluation Completed!")
