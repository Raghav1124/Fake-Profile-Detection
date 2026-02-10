
# 📁 Dataset Information

### Project: Fake Profile Detection using Machine Learning

---

## 📌 Dataset Source

* **Source:** Kaggle
* **Dataset Name:** Fake Profile Detection Dataset (Twitter)
* **Original Dataset Link:** [https://www.kaggle.com/datasets/muhammaddesu/fake-profile-detection](https://www.kaggle.com/datasets/muhammaddesu/fake-profile-detection)
* **Alternative Link:** [https://www.kaggle.com/datasets/jacksoncrow/fake-profile-detection](https://www.kaggle.com/datasets/jacksoncrow/fake-profile-detection)

---

## 📂 Files Included

| File Name | Description | Records | Size |
|-----------|-------------|---------|------|
| `users.csv` | Genuine Twitter user profiles | 1,000 | ~150KB |
| `fusers.csv` | Fake/suspicious Twitter user profiles | 1,000 | ~150KB |
| `sample_data.csv` | Combined sample for demonstration | 200 | ~30KB |

---

## 🧠 Dataset Description

The dataset contains Twitter profile information used for binary classification (genuine vs. fake profiles).

### Features Included:
- **Basic Metadata:**
  - `name`: User's display name
  - `lang`: Language preference (categorical)
  
- **Activity Metrics:**
  - `statuses_count`: Number of tweets/posts
  - `followers_count`: Number of followers
  - `friends_count`: Number of accounts followed
  - `favourites_count`: Number of liked posts
  - `listed_count`: Number of public list memberships

### Target Variable:
- **Binary classification:** 1 = Genuine profile, 0 = Fake profile

### Dataset Statistics:
- **Total Records:** 2,000 (balanced: 1,000 genuine + 1,000 fake)
- **Features:** 7 original features (expanded during preprocessing)
- **Missing Values:** Minimal to none in original dataset

---

## 🔐 Data Usage & Privacy

- ✅ **Public Dataset:** Available on Kaggle for research/educational use
- ✅ **Anonymized:** No personally identifiable information (PII)
- ✅ **Ethical Use:** Intended for academic research in cybersecurity/ML
- ⚠️ **Real-world Caution:** Model should be validated before production deployment
- 🔒 **Privacy Compliant:** Dataset contains only public profile metrics

**Note:** For repository size management, only essential data files are included. Full dataset can be downloaded from Kaggle using the links above.

---

## 🔁 Reproducibility Instructions

### Option 1: Use Included Sample Data
```bash
# The project already includes sample_data.csv for testing
python src/fake_profile_detection_final.py
```

### Option 2: Download Full Dataset
1. Visit the Kaggle dataset link above
2. Download `users.csv` and `fusers.csv`
3. Place both files in the `data/` directory:
```
fake-profile-detection/data/
├── users.csv      # ← Place downloaded file here
├── fusers.csv     # ← Place downloaded file here
└── README.md
```
4. Update file paths in the code if necessary
5. Run the model:
```bash
python src/fake_profile_detection_final.py
```

### Option 3: Use Kaggle API (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d muhammaddesu/fake-profile-detection

# Unzip and place in data/ directory
unzip fake-profile-detection.zip -d data/
```

---

## 📊 Data Preprocessing Pipeline

1. **Data Loading:** Combine `users.csv` (genuine) and `fusers.csv` (fake)
2. **Feature Engineering:**
   - Extract gender from names using `gender-guesser`
   - One-hot encode language feature
   - Generate polynomial features (degree=2)
3. **Train-Test Split:** 80-20 split with stratification
4. **Feature Scaling:** StandardScaler for SVM compatibility

---

## 🧪 Model Performance

Using this dataset with SVM (RBF kernel):
- **Test Accuracy:** 99.11%
- **Precision (Fake):** 98%
- **Recall (Fake):** 100%
- **F1-Score:** 0.99 for both classes

---

## 📎 License & Attribution

- **Dataset License:** [Kaggle Dataset License](https://www.kaggle.com/general/24345)
- **Intended Use:** Academic research, machine learning education
- **Citation Request:** If used in publications, please cite the original Kaggle dataset
- **Modifications:** This dataset has been used for feature engineering and preprocessing as part of the ML pipeline

---

## ⚠️ Important Notes

1. **Dataset Limitations:**
   - Twitter-specific features may not generalize to other platforms
   - Gender detection from names has cultural/regional limitations
   - Dataset may contain sampling biases

2. **Ethical Considerations:**
   - This model is for educational purposes
   - Real-world deployment requires additional validation
   - Consider false positive impacts on genuine users

3. **Maintenance:**
   - Periodically check Kaggle for dataset updates
   - Monitor model performance on new data
   - Consider retraining with fresh data periodically

---

## 🤝 Contributing to Dataset

If you encounter issues with the dataset:
1. Report to the original Kaggle dataset maintainer
2. Open an issue in this repository for preprocessing concerns
3. Consider contributing to open-source dataset alternatives

---

**Happy Coding!** 🚀

*For questions about dataset usage, open an issue or refer to the main project README.*
