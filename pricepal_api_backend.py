# PricePal Backend (API) - Flask Version
# Handles: Loading dataset, answering queries with GPT, forecasting with Prophet, charting

from flask import Flask, request, jsonify
import pandas as pd
import openai
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import requests
from prophet import Prophet
import tempfile
import os

# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # <-- Set this on your deployment
CSV_URL = "https://drive.google.com/uc?export=download&id=1u_XB1Pqn9zaIrcnMq7RluCbItAME2LIL"

# === APP INIT ===
app = Flask(__name__)

# === LOAD DATA ===
@staticmethod
def load_data():
    df = pd.read_csv(CSV_URL)
    df.dropna(subset=['Month', 'State', 'Commodity', 'Price'], inplace=True)
    df['Month'] = pd.to_datetime(df['Month'])
    return df

data = load_data()

# === FORECASTING ===
def forecast_price(df, commodity, state):
    sub = df[(df['Commodity'].str.lower() == commodity.lower()) & (df['State'].str.lower() == state.lower())]
    if len(sub) < 4:
        return None, None

    prophet_df = sub[['Month', 'Price']].rename(columns={"Month": "ds", "Price": "y"})
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    return round(forecast['yhat'].iloc[-1], 2), chart

# === GPT-BASED QUERY HANDLER ===
def generate_answer(query):
    prompt = f"You are a Nigerian market assistant with data on food prices. Answer the question below using specific figures from the dataset if available, or mention if forecast is used.\nQuestion: {query}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for Nigerian food prices."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# === API ROUTE ===
@app.route("/query", methods=["POST"])
def handle_query():
    q = request.json.get("query", "").strip()
    if not q:
        return jsonify({"answer": "Empty query."})

    try:
        # Rough guess of commodity/state from query (basic NLP)
        for commodity in data['Commodity'].unique():
            if commodity.lower() in q.lower():
                selected_commodity = commodity
                break
        else:
            selected_commodity = None

        for state in data['State'].unique():
            if state.lower() in q.lower():
                selected_state = state
                break
        else:
            selected_state = "Lagos"  # Default fallback

        if selected_commodity:
            forecast, chart = forecast_price(data, selected_commodity, selected_state)
        else:
            forecast, chart = None, None

        gpt_answer = generate_answer(q)

        if forecast:
            gpt_answer += f"\nForecasted price for {selected_commodity} in {selected_state}: ₦{forecast}" 

        return jsonify({"answer": gpt_answer, "chart": chart})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
