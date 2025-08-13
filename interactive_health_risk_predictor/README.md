# 🩺 Interactive Health Risk Predictor

An interactive Streamlit app that predicts the risk of diabetes based on health data and explains the **top 3 factors** influencing the prediction using SHAP.

## Features
- User-friendly input form
- Real-time prediction
- SHAP-based explanations
- Risk percentage display

## Tech Stack
- Python
- Streamlit
- scikit-learn
- Matplotlib
- Seaborn
- SHAP

## Dataset
- Source: Kaggle Diabetes Database (included in `/data/diabetes.csv`)

## Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
