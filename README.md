🩺 Interactive Health Risk Predictor

An interactive web application built with Streamlit that predicts the risk of diabetes based on key health indicators. The model is powered by a Random Forest Classifier trained on the Pima Indians Diabetes Dataset, with preprocessing steps like imputation and scaling handled via a pipeline.

🚀 Features

User-friendly interface to input health parameters such as glucose level, BMI, blood pressure, and age.

Real-time risk prediction using a trained machine learning model.

Interactive visualizations for better understanding of health data.

Portable and lightweight — runs directly in the browser with Streamlit.

🧠 Tech Stack

Python 3

Streamlit – web app framework

Scikit-learn – model training & pipeline

Pandas, NumPy – data manipulation

Matplotlib / Seaborn – visualizations

📂 Project Structure
interactive_health_risk_predictor/
│── app.py                  # Streamlit app entry point
│── train_model.py           # Script to train and save the ML model
│── models/
│     └── model.pkl          # Trained ML model
│── data/
│     └── diabetes.csv       # Dataset
│── README.md                # Project documentation


⚡ How to Run Locally

1.Clone the repository: 

git clone https://github.com/yourusername/interactive-health-risk-predictor.git

cd interactive-health-risk-predictor

2. Install dependencies:
   
pip install -r requirements.txt

4. Run the app:

streamlit run app.py

🎯 Use Cases

Health awareness and preventive care.

Quick screening tool for educational or research purposes.

Demonstrates real-world application of ML in healthcare.
