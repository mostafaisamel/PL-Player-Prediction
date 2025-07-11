# 🧠 Premier League Player Performance Prediction

This project aims to predict English Premier League (EPL) players' performance using machine learning techniques. It was developed as a graduation thesis to demonstrate applied knowledge in data science, predictive modeling, and sports analytics.

## 🎯 Objective

The goal of this project is to build a predictive model that can forecast key performance indicators (KPIs) of EPL players — such as goal involvement, match rating, or form — based on historical match and player data.

## 📚 Background

Accurately predicting football player performance has significant applications in sports analytics, fantasy leagues, and club decision-making. Traditional scouting methods are often limited in scope; this project leverages data-driven methods to enhance prediction accuracy and uncover hidden player insights.

## ⚙️ Methodology

1. **Data Collection & Preprocessing**
   - Cleaned missing and inconsistent values
   - Normalized numerical features
   - Encoded categorical variables (position, team, etc.)

2. **Feature Engineering**
   - Derived metrics like goal contribution rate, defensive efficiency
   - Analyzed player form over recent matches

3. **Model Development**
   - Trained and compared multiple classifiers:
     - Logistic Regression
     - Random Forest
     - Support Vector Machines (SVM)
     - XGBoost

4. **Model Evaluation**
   - Metrics used: Accuracy, F1-score, Precision, Recall, ROC-AUC
   - Cross-validation for generalizability

## 📈 Results

> 🥇 **Best Model**: XGBoost achieved the highest performance with an F1-score of **XX%** and ROC-AUC of **XX%** (fill with actual values)

Visualization examples:
- Feature importance
- Confusion matrix
- Model performance comparison

## 🗂️ Project Structure

```
PL-Player-Prediction/
├── data/                # Raw and cleaned datasets
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── models/              # Trained models (optional)
├── plots/               # Performance visualizations
├── pl_predictor.py      # Main ML script
├── requirements.txt     # Project dependencies
└── README.md            # Documentation
```

## 🧪 Technologies Used

- Python 3.10
- pandas, numpy
- scikit-learn
- XGBoost
- seaborn, matplotlib
- Jupyter Notebook

## 🔍 Sample Predictions

(Include a sample table or image showing predicted vs. actual performance.)

## 💡 Future Work

- Expand dataset to multiple seasons or leagues
- Integrate time-series modeling (e.g., LSTM)
- Create a web-based dashboard using Streamlit
- Add model explainability (SHAP / LIME)
- Deploy model with a REST API (FastAPI)

## 📜 Citation

If you use this work or base your project on it, please cite:

```
@project{isamel2025plprediction,
  author    = {Mostafa Isamel},
  title     = {Premier League Player Performance Prediction using Machine Learning},
  year      = {2025},
  note      = {Graduation Project, [Your University Name]}
}
```

## 👨‍💻 Author

**Mostafa Isamel**  
Graduating Data Science Student  
GitHub: [@mostafaisamel](https://github.com/mostafaisamel)

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/mostafaisamel/PL-Player-Prediction.git
cd PL-Player-Prediction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks

```bash
jupyter notebook
```

Explore the notebooks in `/notebooks` to view EDA, model training, and results.

---

## 🤝 Contributions

Feel free to open an issue or submit a pull request for improvements, suggestions, or questions.
