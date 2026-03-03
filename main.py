import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('data/train.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n" + "="*60)
print("STEP 1: HANDLE MISSING VALUES")
print("="*60)

print("\nMissing values BEFORE handling:")
print(df.isnull().sum())

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col] = df[col].fillna(mode_val)

print("\nMissing values AFTER handling:")
print(df.isnull().sum())

print("\n" + "="*60)
print("STEP 2: ENCODE CATEGORICAL COLUMNS")
print("="*60)

# Identify all object columns
object_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {object_cols}")

# Encode ALL categorical columns (including Loan_ID)
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}")

print("\nDataframe after encoding:")
print(df.head())
print("\nDataframe dtypes:")
print(df.dtypes)

print("\n" + "="*60)
print("STEP 3: PREPARE DATA FOR TRAINING")
print("="*60)

# Create feature matrix and target vector
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1).values.astype(float)
y = df['Loan_Status'].values

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Feature columns: {df.drop(['Loan_ID', 'Loan_Status'], axis=1).columns.tolist()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\n" + "="*60)
print("STEP 4: TRAIN MODELS")
print("="*60)

# Model 1: Logistic Regression
print("\n1. Training Logistic Regression...")
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"\n   Logistic Regression Results:")
print(f"   • Accuracy:  {acc_lr:.4f}")
print(f"   • Precision: {prec_lr:.4f}")
print(f"   • Recall:    {rec_lr:.4f}")
print(f"   • F1 Score:  {f1_lr:.4f}")

# Model 2: Random Forest
print("\n2. Training Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"\n   Random Forest Results:")
print(f"   • Accuracy:  {acc_rf:.4f}")
print(f"   • Precision: {prec_rf:.4f}")
print(f"   • Recall:    {rec_rf:.4f}")
print(f"   • F1 Score:  {f1_rf:.4f}")

# Model 3: Support Vector Machine
print("\n3. Training Support Vector Machine (SVM)...")
model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
rec_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"\n   Support Vector Machine Results:")
print(f"   • Accuracy:  {acc_svm:.4f}")
print(f"   • Precision: {prec_svm:.4f}")
print(f"   • Recall:    {rec_svm:.4f}")
print(f"   • F1 Score:  {f1_svm:.4f}")

print("\n" + "="*60)
print("STEP 5: SELECT BEST MODEL")
print("="*60)

models_dict = {
    'Logistic Regression': (acc_lr, y_pred_lr, 'LR'),
    'Random Forest': (acc_rf, y_pred_rf, 'RF'),
    'SVM': (acc_svm, y_pred_svm, 'SVM')
}

best_model_name = max(models_dict, key=lambda x: models_dict[x][0])
best_accuracy = models_dict[best_model_name][0]
y_pred_best = models_dict[best_model_name][1]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"✓ Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

print("\n" + "="*60)
print("STEP 6: DETAILED RESULTS")
print("="*60)

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=['Approved', 'Rejected']))

cm_best = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm_best)
print(f"\nTrue Negatives:  {cm_best[0, 0]}")
print(f"False Positives: {cm_best[0, 1]}")
print(f"False Negatives: {cm_best[1, 0]}")
print(f"True Positives:  {cm_best[1, 1]}")

print("\n" + "="*60)
print("STEP 7: SAMPLE PREDICTIONS")
print("="*60)

print("\nSample Predictions (First 5 test samples):")
for i in range(min(5, len(y_test))):
    actual = "Approved" if y_test[i] == 0 else "Rejected"
    predicted = "Approved" if y_pred_best[i] == 0 else "Rejected"
    match = "✓" if y_test[i] == y_pred_best[i] else "✗"
    print(f"   Sample {i+1}: Actual = {actual:10s} | Predicted = {predicted:10s} {match}")

print("\n" + "="*60)
print("STEP 8: CREATE VISUALIZATIONS")
print("="*60)

# Create confusion matrix visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
im0 = axes[0].imshow(cm_lr, cmap='Blues', interpolation='nearest')
axes[0].set_title(f'Logistic Regression\nAccuracy: {acc_lr:.4f}', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=10)
axes[0].set_xlabel('Predicted', fontsize=10)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Approved', 'Rejected'])
axes[0].set_yticklabels(['Approved', 'Rejected'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm_lr[i, j]), ha='center', va='center', 
                    color='white' if cm_lr[i, j] > cm_lr.max()/2 else 'black', 
                    fontsize=14, fontweight='bold')
plt.colorbar(im0, ax=axes[0])

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
im1 = axes[1].imshow(cm_rf, cmap='Greens', interpolation='nearest')
axes[1].set_title(f'Random Forest\nAccuracy: {acc_rf:.4f}', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=10)
axes[1].set_xlabel('Predicted', fontsize=10)
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Approved', 'Rejected'])
axes[1].set_yticklabels(['Approved', 'Rejected'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm_rf[i, j]), ha='center', va='center', 
                    color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black', 
                    fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=axes[1])

# SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
im2 = axes[2].imshow(cm_svm, cmap='Oranges', interpolation='nearest')
axes[2].set_title(f'Support Vector Machine\nAccuracy: {acc_svm:.4f}', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Actual', fontsize=10)
axes[2].set_xlabel('Predicted', fontsize=10)
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(['Approved', 'Rejected'])
axes[2].set_yticklabels(['Approved', 'Rejected'])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm_svm[i, j]), ha='center', va='center', 
                    color='white' if cm_svm[i, j] > cm_svm.max()/2 else 'black', 
                    fontsize=14, fontweight='bold')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Graph saved as 'model_results.png'")
plt.show()

print("\n" + "="*60)
print("PROJECT COMPLETE!")
print("="*60)