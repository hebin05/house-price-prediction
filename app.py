from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

saved_model = joblib.load("sport_car_price_model.pkl")
model = saved_model["model"]
columns = saved_model["columns"]

# Automatically get all car makes and models

car_makes = sorted([
    col.replace("Car Make_", "")
    for col in columns
    if col.startswith("Car Make_")
])

car_models = sorted([
    col.replace("Car Model_", "")
    for col in columns
    if col.startswith("Car Model_")
])


@app.route("/")
def home():
    return render_template(
        "index.html",
        car_makes=car_makes,
        car_models=car_models
    )


@app.route("/predict", methods=["POST"])
def predict():

    horsepower = float(request.form["horsepower"])
    torque = float(request.form["torque"])
    car_make = request.form["car_make"]
    car_model = request.form["car_model"]

    new_data = pd.DataFrame(0, index=[0], columns=columns)

    new_data["Horsepower"] = horsepower
    new_data["Torque (lb-ft)"] = torque

    make_col = "Car Make_" + car_make
    model_col = "Car Model_" + car_model

    if make_col in columns:
        new_data[make_col] = 1

    if model_col in columns:
        new_data[model_col] = 1

   # Predict price

    prediction = model.predict(new_data)

    usd_price = prediction[0]

    # USD to INR conversion
    exchange_rate = 95.39
    inr_price = usd_price * exchange_rate

    return render_template(
        "index.html",
        usd_price="${:,.2f}".format(usd_price),
        inr_price="₹{:,.2f}".format(inr_price),
        car_makes=car_makes,
        car_models=car_models
    )


if __name__ == "__main__":
    app.run(debug=True)