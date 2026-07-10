# 🏡 Real Estate Valuation Engine

A full-stack machine learning application that predicts house prices based on property features. Built with a Python/scikit-learn backend and an interactive Streamlit frontend.

---

## 🚀 Overview

This project ingests raw real estate data, engineers meaningful features (such as combining room counts and frequency-encoding ZIP codes), trains multiple regression models, and serves the best-performing model through a user-friendly web interface.

### ✨ Key Features

- **Automated Data Cleaning:** Handles missing values and removes duplicate entries.
- **Feature Engineering:** Converts geographical categorical data (ZIP codes) into predictive frequency-encoded features.
- **Model Training:** Compares Linear Regression and Random Forest Regressor models to determine the best performer.
- **Interactive UI:** A responsive, custom-styled Streamlit dashboard for real-time house price predictions.

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.10+ |
| **Machine Learning** | scikit-learn, pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Frontend** | Streamlit |
| **Model Serialization** | Joblib |

---

## 📂 Project Structure

```text
AlgoHub_Task1_HousePricePrediction/
│
├── data/                  # Raw and processed datasets
├── figures/               # EDA visualizations
├── models/                # Saved models and evaluation metrics
├── src/
│   ├── preprocess.py          # Data cleaning
│   ├── feature_engineering.py # Feature engineering
│   ├── train.py               # Model training
│   └── evaluate.py            # Model evaluation
│
├── assets/                # UI screenshots for documentation
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
└── README.md
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/omaimasiddiqui/AlgoHub_Task1_HousePricePrediction.git
cd AlgoHub_Task1_HousePricePrediction
```

### 2️⃣ Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ (Optional) Run the Training Pipeline

The repository includes pre-trained models, so this step is optional.

```bash
python src/preprocess.py
python src/feature_engineering.py
python src/train.py
```

### 5️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Model Evaluation

The models were evaluated using an **80/20 train-test split**.

| Model | R² Score |
|------|---------:|
| Linear Regression | **~0.579** |
| Random Forest Regressor | ~0.565 |

### 🏆 Best Performing Model

**Linear Regression** was selected for deployment based on its superior **R² Score** and **Mean Absolute Error (MAE)**.

---

## 💡 Future Improvements

- Hyperparameter tuning using **GridSearchCV** or **RandomizedSearchCV**
- Support for additional regression models (XGBoost, LightGBM, CatBoost)
- Deployment on Streamlit Cloud or Render
- Improved feature engineering and automated model retraining
- Interactive visualizations for prediction insights

---


## 📄 License

This project is intended for educational and learning purposes.