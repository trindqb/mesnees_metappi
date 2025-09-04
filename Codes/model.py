import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import numpy as np
import pandas as pd


# Separate features and labels
X = df.drop('Label', axis=1)
y = df['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = LogisticRegression(max_iter=500,random_state=42)

# Set up 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

# Cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    # Store
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    aucs.append(auc)

    print(f"Fold {fold}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}\n")

# Final average results
avg_metrics = {
    "Accuracy": (np.mean(accuracies), np.std(accuracies)),
    "Precision": (np.mean(precisions), np.std(precisions)),
    "Recall": (np.mean(recalls), np.std(recalls)),
    "F1-Score": (np.mean(f1_scores), np.std(f1_scores)),
    "ROC-AUC": (np.mean(aucs), np.std(aucs)),
}
print("5-Fold Cross-Validation Results :")
print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"ROC-AUC: {np.mean(aucs):.4f}(±{ np.std(aucs):.4f})" )



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Set up 5-fold cross-validation (stratified to maintain class balance)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
aucs = []

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    # Split data into training and test sets for this fold
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    aucs.append(auc)

    # Print fold results
    print(f"Fold {fold}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}\n")

# Calculate and print average metrics and standard deviations
print("5-Fold Cross-Validation Results (RandomForestClassifier):")
print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"ROC-AUC: {np.mean(aucs):.4f}(±{ np.std(aucs):.4f})" )


from xgboost import XGBClassifier
import numpy as np

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the XGBoost model
model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

# Set up 5-fold cross-validation (stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []
aucs = []

# 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    aucs.append(auc)

    print(f"Fold {fold}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}\n")

# Print average metrics
print("5-Fold Cross-Validation Results (XGBoost):")
print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"Average ROC-AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")




# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the LightGBM model
model = LGBMClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Set up 5-fold cross-validation (stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []
aucs = []

# 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    aucs.append(auc)

    print(f"Fold {fold}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}\n")

# Print average metrics
print("5-Fold Cross-Validation Results (LightGBM):")
print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
print(f"Average ROC-AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up 5-fold cross-validation (stratified to maintain class balance)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    # Split data into training and test sets for this fold
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Define the neural network model
    model_nn = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model_nn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

    # Make predictions
    y_pred_proba = model_nn.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Print fold results
    print(f"Fold {fold}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}\n")

# Calculate and print average metrics and standard deviations
print("5-Fold Cross-Validation Results (Neural Network):")
print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")