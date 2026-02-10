# Fake Profile Detection using Machine Learning (SVM)

This project aims to detect fake profiles using machine learning techniques, specifically employing Support Vector Machines (SVM) with feature engineering and hyperparameter tuning. The model achieves **99.11% accuracy** in distinguishing between genuine and fake Twitter profiles.

## 📁 Project Structure

```
fake-profile-detection/
│
├── notebooks/
│   ├── fake_profile_detection_v1.ipynb     # Initial exploratory notebook
│   └── fake_profile_detection_final.ipynb  # Final implementation notebook
│
├── src/
│   └── fake_profile_detection_final.py     # Production-ready Python script
│
├── data/
│   ├── users.csv                           # Genuine user profiles dataset
│   ├── fusers.csv                          # Fake user profiles dataset
│   └── README.md                           # Data description and source info
│
├── README.md                               # This file
├── requirements.txt                        # Project dependencies
└── .gitignore                              # Git ignore file
```

## 📊 Dataset

The dataset consists of two CSV files:
- **users.csv**: Genuine Twitter user profiles (1,000 records)
- **fusers.csv**: Fake/suspicious Twitter user profiles (1,000 records)

### Dataset Features:
- `name`: User's display name
- `statuses_count`: Number of tweets/posts
- `followers_count`: Number of followers
- `friends_count`: Number of friends/following
- `favourites_count`: Number of liked posts
- `listed_count`: Number of public list memberships
- `lang`: Language preference

**Note**: The full dataset is available on Kaggle and is included in this repository for convenience.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-profile-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Additional gender detection libraries:
```bash
pip install sexmachine
pip install gender-guesser
```

### Usage

#### Jupyter Notebooks
- Open `notebooks/fake_profile_detection_final.ipynb` for the complete analysis
- The notebook includes data loading, preprocessing, feature engineering, model training, and evaluation

#### Python Script
Run the production script:
```bash
python src/fake_profile_detection_final.py
```

## 🔧 Features

### 1. **Data Preprocessing**
- Concatenation of genuine and fake user datasets
- Label encoding (1 for genuine, 0 for fake)
- Gender extraction from names using `gender-guesser`
- Language encoding using one-hot encoding

### 2. **Feature Engineering**
- **Selected Features:**
  - `statuses_count`
  - `followers_count`
  - `friends_count`
  - `favourites_count`
  - `listed_count`
  - `gender` (extracted from names)
- Gender mapping: Female (1), Male (0), Unknown (-1)
- Polynomial features (degree=2) for capturing feature interactions

### 3. **Model Training**
- **Algorithm:** Support Vector Machine (SVM) with RBF kernel
- **Hyperparameter Tuning:**
  - GridSearchCV with 5-fold stratified cross-validation
  - Parameters: `C` [0.1, 1, 10, 100], `gamma` ['scale', 'auto', 0.01, 0.001]
- **Standardization:** Features scaled using StandardScaler
- **Train-Test Split:** 80-20 split with stratification

### 4. **Evaluation Metrics**
- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve and AUC Score

## 📈 Model Performance

### **Test Results:**
- **Accuracy:** **99.11%**
- **AUC Score:** ~0.99

### **Detailed Classification Report:**
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       268
           1       1.00      0.98      0.99       296

    accuracy                           0.99       564
   macro avg       0.99      0.99      0.99       564
weighted avg       0.99      0.99      0.99       564
```

### **Key Performance Indicators:**
- **Fake Profiles (Class 0):** 98% precision, 100% recall
- **Genuine Profiles (Class 1):** 100% precision, 98% recall
- **Overall:** Excellent balance between precision and recall for both classes

## 🛠️ Dependencies

Key libraries used:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `gender-guesser` - Gender detection from names
- `matplotlib` - Data visualization

See `requirements.txt` for complete list.

## 🔍 Key Insights

1. **High Accuracy:** The model achieves 99.11% accuracy, demonstrating strong capability in distinguishing fake profiles
2. **Feature Importance:** Social metrics like followers_count and friends_count are strong indicators of profile authenticity
3. **Gender Feature:** Adding gender information (extracted from names) improves model performance
4. **Polynomial Features:** Capturing feature interactions enhances model capability
5. **SVM Performance:** SVM with RBF kernel and proper hyperparameter tuning performs exceptionally well for this binary classification problem
6. **Balanced Performance:** Both precision and recall are high for both classes, minimizing false positives and false negatives

## 📝 Usage Notes

1. The dataset is included in the `data/` folder for immediate use
2. The model is trained on Twitter profile data - adapt feature selection for other platforms
3. Gender detection relies on first names; consider cultural naming variations
4. Adjust hyperparameters based on your specific requirements
5. Consider feature scaling importance for SVM models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Potential areas for improvement:
- Experiment with other ML algorithms (Random Forest, XGBoost, Neural Networks)
- Add more sophisticated feature engineering
- Implement real-time prediction API
- Create a user-friendly web interface

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Contributors to the `gender-guesser` library
- Scikit-learn community for comprehensive ML tools
- Twitter for the public profile data
