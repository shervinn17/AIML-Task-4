# Logistic Regression Classification Project

## 🔍 Objective
Build a binary classifier using **Logistic Regression** to predict whether a tumor is malignant or benign based on diagnostic features.

## 📁 Dataset
The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains features computed from digitized images of breast mass FNA (fine needle aspirates).

- Rows: 569
- Features: 30 real-valued inputs
- Target: `diagnosis` (M = Malignant, B = Benign)

## 🛠️ Tools & Libraries
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 📌 Project Steps

### 1. Load and Prepare Data
- Dropped unnecessary columns like `id` and `Unnamed: 32`
- Encoded `diagnosis` as binary (`M` = 1, `B` = 0)

### 2. Train-Test Split
- Data split into 80% training and 20% testing using `train_test_split`

### 3. Feature Standardization
- Standardized features using `StandardScaler` to normalize scales

### 4. Model Training
- Used `LogisticRegression` from `sklearn` to fit the model on training data

### 5. Model Evaluation
- Evaluated the model with:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - ROC-AUC Score and ROC Curve

### 6. Threshold Tuning & Sigmoid Function
- Visualized the sigmoid function to understand logistic output
- Explained how decision threshold can impact classification

## 📊 Results
- Achieved high ROC-AUC score indicating strong model performance
- Clear separation between malignant and benign predictions

## 📈 Visualizations
- ROC Curve to assess classifier performance
- Sigmoid function to understand logistic regression output behavior

## 🔁 Improvements
- Cross-validation for more robust evaluation
- Feature selection to reduce dimensionality
- GridSearchCV for hyperparameter tuning

---

## 🧠 Concept: Sigmoid Function
The logistic regression model uses the **sigmoid function** to map predicted values to probabilities:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
