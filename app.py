from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "OverallQual": float(request.form["OverallQual"]),
        "GrLivArea": float(request.form["GrLivArea"]),
        "GarageCars": float(request.form["GarageCars"]),
        "GarageArea": float(request.form["GarageArea"]),
        "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
        "1stFlrSF": float(request.form["FirstFlrSF"]),
        "FullBath": float(request.form["FullBath"]),
        "TotRmsAbvGrd": float(request.form["TotRmsAbvGrd"]),
        "YearBuilt": float(request.form["YearBuilt"])
    }

    features = pd.DataFrame([data])

    # Create all training columns
    final_df = pd.DataFrame(columns=model_columns)

    # Append user data
    final_df = pd.concat([final_df, features], ignore_index=True)

    # Missing columns become 0
    final_df = final_df.fillna(0)

    # Keep exact training order
    final_df = final_df[model_columns]

    prediction = model.predict(final_df)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ${prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)