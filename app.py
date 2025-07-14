from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("gbc_model_clean.pkl")

# Load and clean dataset
df = pd.read_csv("pistachio.csv")
df.columns = df.columns.str.strip()

FEATURE_NAMES = [
    'AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS',
    'ECCENTRICITY', 'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA',
    'EXTENT', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS',
    'SHAPEFACTOR_1', 'SHAPEFACTOR_2', 'SHAPEFACTOR_3', 'SHAPEFACTOR_4'
]

class_map = {0: "Kirmizi_Pistachio", 1: "Siirt_Pistachio"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    sample_input = {name: "" for name in FEATURE_NAMES}
    active_action = None

    if request.method == "POST":
        active_action = request.form.get("action")

        if active_action == "suggest":
            random_row = df.sample(1).iloc[0]
            sample_input = {feature: round(random_row[feature], 4) for feature in FEATURE_NAMES}

        elif active_action == "predict":
            try:
                input_values = [float(request.form.get(name)) for name in FEATURE_NAMES]
                input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

                result = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
                confidence = round(max(prob) * 100, 2)
                prediction = class_map.get(result, result)

                sample_input = {name: request.form.get(name) for name in FEATURE_NAMES}

            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template("index.html",
                           features=FEATURE_NAMES,
                           sample_input=sample_input,
                           prediction=prediction,
                           confidence=confidence,
                           active_action=active_action)

if __name__ == "__main__":
    app.run(debug=True)
