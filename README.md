# 🧠 Bank Term Deposit Prediction with Decision Tree Classifier

This project uses a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit based on their **demographic** and **behavioral data**. The analysis includes model training, evaluation, and visualization of both the decision tree and performance metrics.

---

## 📁 Dataset

- **bank_full.csv** – Full dataset containing customer information and subscription outcome.
- Source: [Bank Marketing Data Set (UCI)](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## 🔧 Features Used

Includes variables such as:
- Age, Job, Marital Status, Education
- Credit default, Balance, Housing/Purchase Loans
- Contact type, Month, Day, Duration of last call

The target variable is:
- `y`: Whether the client subscribed to a term deposit (`yes` or `no`)

---

## 📊 Visual Outputs

All visualizations are saved under the `images/` directory:

| File Name              | Description                                 |
|------------------------|---------------------------------------------|
| `confusion_matrix.png` | Confusion matrix of prediction results      |
| `decision_tree.png`    | Full visualization of the trained decision tree |

---

## ✅ Model Summary

- Model: `DecisionTreeClassifier` from scikit-learn
- Max depth: `4` (to prevent overfitting and improve interpretability)
- Evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - Decision tree rule set

---

## 📂 Output

- Trained model evaluation printed to console
- Decision tree and confusion matrix saved as `.png` images
- Human-readable decision rules printed via `export_text()`

---

## 🚀 How to Run

### 1. Clone this Repository

```bash
git clone https://github.com/YourUsername/PRODIGY_DS_03.git
cd PRODIGY_DS_03
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 3. Run the Script

```bash
python bank_term_prediction.py
```

> Make sure `bank_full.csv` is in the same directory as the script.


---

## 📧 Author

**Souvik Tikader**
GitHub: [@SouvikTikader](https://github.com/SouvikTikader)


