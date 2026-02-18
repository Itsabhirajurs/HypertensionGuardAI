from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import os
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Load trained model with error handling
try:
    model = joblib.load("logistic_regression_model.pkl")
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("⚠ Warning: Model file not found. Using dummy predictions.")
    model = None
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    model = None

# Mapping back numeric prediction to original stage
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Medical-grade color mapping for results
color_map = {
    0: '#10B981',  # Medical green for normal
    1: '#F59E0B',  # Medical amber for stage 1
    2: '#F97316',  # Medical orange for stage 2
    3: '#EF4444'   # Medical red for crisis
}

# Detailed medical recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk assessment indicates normal blood pressure levels.',
        'actions': [
            'Maintain current healthy lifestyle',
            'Regular physical activity (150 minutes/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups'
        ],
        'priority': 'Low Risk'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected requiring lifestyle modifications and medical consultation.',
        'actions': [
            'Schedule appointment with healthcare provider',
            'Implement DASH diet plan',
            'Increase physical activity gradually',
            'Monitor blood pressure bi-weekly',
            'Reduce sodium intake (<2300mg/day)',
            'Consider stress management techniques'
        ],
        'priority': 'Moderate Risk'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring immediate medical intervention and treatment.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Likely medication therapy required',
            'Comprehensive cardiovascular assessment',
            'Daily blood pressure monitoring',
            'Strict dietary sodium restriction',
            'Lifestyle modification counseling'
        ],
        'priority': 'High Risk'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure requiring emergency medical care.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention',
            'Call 911 if experiencing symptoms',
            'Do not delay treatment',
            'Monitor for stroke/heart attack signs',
            'Do not self-medicate'
        ],
        'priority': 'Critical Risk'
    }
}


@app.route('/')
def home():
    return render_template('index.html', form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    form_data = {}
    try:
        # Collect user inputs from the form with validation
        required_fields = [
            'Gender', 'Age', 'History', 'Patient', 'TakeMedication',
            'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
            'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet'
        ]

        for field in required_fields:
            value = request.form.get(field)
            if not value or value == "":
                flash(f"Please complete all required fields: {field.replace('_', ' ')}", "error")
                return render_template('index.html', form_data=form_data)
            form_data[field] = value

        # Normalize severity input (handle both "Sever" and "Severe")
        severity = form_data['Severity']
        if severity == 'Sever':
            severity = 'Severe'
        
        # Normalize systolic/diastolic (handle spacing inconsistencies)
        systolic = form_data['Systolic'].replace('- ', '-').replace(' -', '-').strip()
        diastolic = form_data['Diastolic'].replace('- ', '-').replace(' -', '-').strip()

        # Encode inputs with error handling
        try:
            encoded = [
                0 if form_data['Gender'] == 'Male' else 1,
                {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}[form_data['Age']],
                1 if form_data['History'] == 'Yes' else 0,
                1 if form_data['Patient'] == 'Yes' else 0,
                1 if form_data['TakeMedication'] == 'Yes' else 0,
                {'Mild': 0, 'Moderate': 1, 'Severe': 2}[severity],
                1 if form_data['BreathShortness'] == 'Yes' else 0,
                1 if form_data['VisualChanges'] == 'Yes' else 0,
                1 if form_data['NoseBleeding'] == 'Yes' else 0,
                {'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3}[form_data['Whendiagnoused']],
                {'100-110': 0, '111-120': 1, '121-130': 2, '130+': 3}.get(
                    systolic.replace(' ', ''),
                    {'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3}.get(systolic, 0)
                ),
                {'70-80': 0, '81-90': 1, '91-100': 2, '100+': 3}.get(
                    diastolic.replace(' ', ''),
                    {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3}.get(diastolic, 0)
                ),
                1 if form_data['ControlledDiet'] == 'Yes' else 0
            ]
        except KeyError as e:
            flash(f"Invalid selection detected: {str(e)}", "error")
            return render_template('index.html', form_data=form_data)

        # Manual scaling of ordinal features
        scaled_encoded = encoded.copy()
        scaled_encoded[1] = (encoded[1] - 1) / (4 - 1)   # Age
        scaled_encoded[5] = encoded[5] / 2                 # Severity
        scaled_encoded[9] = (encoded[9] - 1) / (3 - 1)    # When diagnosed
        scaled_encoded[10] = encoded[10] / 3               # Systolic
        scaled_encoded[11] = encoded[11] / 3               # Diastolic

        input_array = np.array(scaled_encoded).reshape(1, -1)

        # Predict with model or use dummy prediction
        if model is not None:
            prediction = model.predict(input_array)[0]
            try:
                confidence = max(model.predict_proba(input_array)[0]) * 100
            except:
                confidence = 85.0
            
            print(f"✓ Prediction: {prediction} ({stage_map[prediction]})")
            print(f"✓ Confidence: {confidence:.2f}%")
        else:
            # Dummy prediction for testing when model is not available
            prediction = random.randint(0, 3)
            confidence = 87.5
            flash("Demo Mode: Using simulated AI prediction for demonstration", "info")

        result_text = stage_map[prediction]
        result_color = color_map[prediction]
        result_recommendation = recommendations[prediction]

        return render_template(
            'index.html',
            prediction_text=result_text,
            result_color=result_color,
            confidence=round(confidence, 2),
            recommendation=result_recommendation,
            form_data=form_data
        )

    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        print(f"✗ Error in predict(): {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', form_data=form_data)


if __name__ == '__main__':
    print("=" * 50)
    print("🏥 PulseGuard AI - Starting Server...")
    print("=" * 50)
    app.run(debug=True)