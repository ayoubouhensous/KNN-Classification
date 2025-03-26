# K-Nearest Neighbors (KNN) Classifier for Breast Cancer Detection

## Overview
This project implements a **K-Nearest Neighbors (KNN) classifier** to detect breast cancer using the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`. The model is trained and evaluated using different distance metrics, with **hyperparameter tuning** (choosing the best `k` using cross-validation).

## Features
- Loads the **Breast Cancer Wisconsin dataset**.
- **Preprocesses** data using **Standardization** and **Normalization**.
- **Splits** the data into training and test sets.
- Finds the **optimal value of k** using **cross-validation**.
- Evaluates the KNN model using **Accuracy Score** and **Confusion Matrix**.

## Requirements
Make sure you have the required dependencies installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
Run the script using Python:

```bash
python knn_breast_cancer.py
```

## Implementation Details
### 1️⃣ Load Dataset
The **Breast Cancer Wisconsin dataset** is loaded using `sklearn.datasets.load_breast_cancer()`, and then converted into a Pandas DataFrame for easier manipulation.

### 2️⃣ Data Preprocessing
- **Normalization**: Uses `Normalizer()` to scale feature vectors.
- **Standardization**: Uses `StandardScaler()` to transform features to have a mean of 0 and a variance of 1.

### 3️⃣ Hyperparameter Tuning
- The script tests **odd values of k** (from `1` to `49`) using **10-fold cross-validation**.
- The best `k` is chosen based on **minimum error (1 - accuracy score)**.

### 4️⃣ Train & Test KNN Classifier
- The KNN model is trained using the best `k` with the **Manhattan distance metric**.
- Predictions are made on the test set.

### 5️⃣ Model Evaluation
- **Accuracy Score**: Measures overall model performance.
- **Confusion Matrix**: Visualized using Seaborn to analyze model predictions.

## Example Output
```
Le nombre optimal de voisins est 5.
Accuracy: 0.9649
```

## Results
The model achieves **high accuracy (>95%)**, showing that KNN is effective for breast cancer detection when combined with proper preprocessing and hyperparameter tuning.

## Future Improvements
- Experiment with different distance metrics (e.g., Minkowski, Chebyshev).
- Optimize feature selection.
- Compare KNN with other classifiers (e.g., SVM, Random Forest).

## Author
**Ayoub OUHENSOUS**

## License
This project is licensed under the MIT License.
