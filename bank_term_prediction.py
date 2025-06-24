# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.tree import export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Folder for images
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("bank_full.csv", sep=';')

# Clean column names
df.columns = df.columns.str.strip()

# Encode categorical variables
label_encoders = {}
df_encoded = df.copy()
df_encoded.columns = df_encoded.columns.str.strip()

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    
print("Label mapping for y:", label_encoders['y'].classes_)


# Define features (X) and target (y)
X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png",bbox_inches='tight')
plt.close()


# Visualize and save the decision tree
plt.figure(figsize=(36, 12))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=label_encoders['y'].classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Classifier for Term Deposit Prediction", fontsize=20)

# Save 
tree_path = os.path.join(output_dir, "decision_tree.png")
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
plt.close()

# rules
rules = export_text(clf, feature_names=list(X.columns))
print(rules)

