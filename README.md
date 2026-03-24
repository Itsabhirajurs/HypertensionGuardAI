# HypertensionGuard AI 

An AI-powered app that predicts hypertension risk using machine learning. Just fill in your health info and get instant recommendations.

---

## Why I Built This

I wanted to understand how machine learning actually works in real-world applications. Hypertension is a common health issue that many people don't know they have, so I thought it would be cool to build something that helps predict risk early.

Plus, it's a great way to learn full-stack development - from training ML models to deploying live web apps.

---

## What It Does

- **Fill a simple form** with your health details (blood pressure, age, symptoms, etc.)
- **Get instant prediction** - Normal, Stage-1, Stage-2, or Crisis level
- **See personalized recommendations** based on your risk level
- **Learn about prevention** - Tips for healthy blood pressure
- **Understand your numbers** - Blood pressure guidelines explained

---

## Tech Stack

**Backend:**
- Python + Flask (web framework)
- Scikit-learn (machine learning)
- Logistic Regression model

**Frontend:**
- HTML5, CSS3, JavaScript
- Responsive design (works on mobile/desktop)

**Deployment:**
- Render.com (hosting)
- GitHub (version control)

---

## Dataset & Model

- **Training data:** 1,800+ patient records
- **Features used:** 13 medical parameters (age, BP readings, symptoms, etc.)
- **Model:** Logistic Regression
- **Accuracy:** ~87%

The model looks at factors like age, family history, blood pressure readings, and symptoms to make predictions.

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Itsabhirajurs/HypertensionGuardAI.git
cd HypertensionGuardAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open browser
http://127.0.0.1:5000

HypertensionGuardAI/
├── app.py                    # Main app (Flask + ML logic)
├── index.html      # Frontend
├── logistic_regression_model.pkl  # Trained model
├── patient_data.csv          # Training data
├── requirements.txt          # Dependencies
└── Eda/                      # Data analysis notebooks

Features
- Real-time predictions using ML
- 4 risk level classifications (color-coded)
- Personalized health recommendations
- Educational content (guidelines, prevention tips)
- Form validation & error handling
- Mobile-responsive design
- Fast & accurate predictions

Live Demo
Access here: https://hypertensionguard-ai.onrender.com/

Try it out! Fill the form with your data and see the prediction.

Disclaimer 
This is NOT a medical diagnosis tool. It's built for learning purposes only. Always consult a real doctor for health issues. If you have chest pain or breathing problems, call 911.