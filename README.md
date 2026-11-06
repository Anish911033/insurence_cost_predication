# 🏥 Medical Insurance Cost Prediction

An end-to-end machine learning project that estimates an individual's yearly medical insurance charges from demographic and health details, deployed as a live web app.

🔗 **Live App:** https://insurence-cost-predication.onrender.com/

📊 **Dataset:** [medical_insurance.csv](https://github.com/Anish911033/insurence_cost_predication/blob/main/medical_insurance.csv)

> ⏳ The app is hosted on Render's free plan, so it sleeps after inactivity. The first load may take 30–60 seconds.

---

## 📌 Project Overview

The goal is to understand which factors drive healthcare costs and to build a regression model that predicts expected charges accurately. The workflow covers data cleaning, exploratory analysis, feature engineering, model comparison, and deployment.

## 📂 Dataset

| Feature | Description |
| :--- | :--- |
| `age` | Age of the beneficiary |
| `sex` | `female` / `male` |
| `bmi` | Body Mass Index |
| `children` | Number of dependents |
| `smoker` | `yes` / `no` |
| `region` | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | Yearly medical insurance charges in USD (**target**) |

The file has 2,772 records. After removing **1,435 duplicates**, **1,337 unique records** remain.

## 🔬 Methodology

1. **Data cleaning:** removed duplicate rows.
2. **EDA:** analysed distributions, skewness and outliers (IQR method), and compared charges across smoker status and BMI categories.
3. **Feature engineering:**
   - Applied a `log1p` transform to the target. Skewness dropped from **1.515** to **-0.090**.
   - One-hot encoded `sex`, `smoker` and `region` (`drop_first=True`).
4. **Modeling:** 80/20 train-test split (`random_state=42`). Predictions are converted back to USD with `expm1` before evaluation.
5. **Deployment:** the best model is saved with `joblib` and served through a Streamlit app on Render.

## 💡 Key Insights

- **Smoking is the biggest cost driver.** Smokers pay about **$32,050** on average versus about **$8,440** for non-smokers.
- **BMI matters.** Obese individuals average about **$15,559**, noticeably higher than the underweight, normal and overweight groups.

## 🤖 Model Results

| Model | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** ✅ | **$2,014.07** | **$4,244.16** | **0.9020** |
| HistGradientBoosting Regressor | $2,372.37 | $4,621.48 | 0.8838 |
| Linear Regression | $3,755.92 | $7,197.03 | 0.7181 |

The **Random Forest** model performed best and explains about **90.2%** of the variance in charges, with an average error of about **$2,014** on unseen data.

## 🗂️ Project Structure

```
├── app.py                         # Streamlit web app
├── medical_cost_predication.ipynb # EDA, training and evaluation
├── medical_insurance.csv          # Dataset
├── insurence_model.pkl            # Trained Random Forest model
├── model_features.pkl             # Feature column order used at inference
├── requirements.txt               # Python dependencies
├── render.yaml                    # Render deployment config
└── README.md
```

## 🚀 Run Locally

```bash
git clone https://github.com/Anish911033/insurence_cost_predication.git
cd insurence_cost_predication
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deployment (Render)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- **Python version:** set with the `PYTHON_VERSION` environment variable

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Streamlit · Render

## ⚠️ Disclaimer

This project is for educational purposes. Predictions are statistical estimates and not real insurance quotes.

## 👤 Author

**Anish** · [GitHub](https://github.com/Anish911033)
