# === inventory_forecast/app.py (FINAL VERSION - SERVES FRONTEND) ===
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import datetime
import math
import os
import csv

# Import your InventoryForecaster class
from utils.forecast import InventoryForecaster

app = Flask(__name__)
CORS(app) # CORS is still good practice

# --- GLOBAL INITIALIZATION ---
forecaster = None
try:
    forecaster = InventoryForecaster()
    print("Flask app: InventoryForecaster initialized, connected to database.")
except Exception as e:
    print(f"Flask app: ERROR initializing Forecaster or connecting to DB: {e}")
    forecaster = None

# --- Mock Product Details ---
def get_product_details(item_id: str):
    mock_products = {
        "FOODS_1_001": {"base_price": 10.0, "expiry_days": 14, "weight_kg": 0.5},
        "FOODS_1_002": {"base_price": 12.0, "expiry_days": 21, "weight_kg": 0.6},
        "HOUSEHOLD_1_001": {"base_price": 5.0, "expiry_days": 365, "weight_kg": 0.2},
        "HOBBIES_1_001": {"base_price": 7.5, "expiry_days": 180, "weight_kg": 0.3},
        "FOODS_1_005": {"base_price": 11.0, "expiry_days": 15, "weight_kg": 0.52},
        "FOODS_2_010": {"base_price": 15.0, "expiry_days": 10, "weight_kg": 0.65},
        "HOUSEHOLD_2_005": {"base_price": 6.5, "expiry_days": 300, "weight_kg": 0.25},
        "HOBBIES_1_002": {"base_price": 8.0, "expiry_days": 170, "weight_kg": 0.31},
        "HOBBIES_2_001": {"base_price": 9.0, "expiry_days": 200, "weight_kg": 0.4},
    }
    return mock_products.get(item_id, {"base_price": 10.0, "expiry_days": 30, "weight_kg": 0.5})

# --- Mock CO2 Estimation ---
def estimate_co2(qty: float, weight_kg: float):
    return qty * weight_kg * (0.5 + 1.0 + 0.1)

# --- Sustainability Log Path ---
SUSTAINABILITY_LOG_PATH = os.path.join(os.path.dirname(__file__), 'sustainability.csv')

# --- Routes ---

# === THIS IS THE KEY CHANGE TO SERVE THE FRONTEND ===
@app.route('/')
def home():
    # Instead of returning text, it now renders your HTML file.
    # Flask automatically looks in the 'templates' folder.
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    if forecaster is None:
        return jsonify({"error": "Forecasting service not ready. Check server logs."}), 500

    data = request.json
    store_id, product_id, fc_date_str = data.get('store_id'), data.get('product_id'), data.get('fc_date')

    if not all([store_id, product_id, fc_date_str]):
        return jsonify({"error": "Missing input: store_id, product_id, and fc_date are required."}), 400

    try:
        fc_date = datetime.datetime.strptime(fc_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    relevant_sales_history = forecaster._get_relevant_sales_history(store_id, product_id, fc_date)
    
    if relevant_sales_history.empty or len(relevant_sales_history) < 14:
        return jsonify({"error": "Insufficient historical data for this item/store. Need at least 14 prior sales days."}), 400

    forecast_result = forecaster.forecast_quantity(store_id, product_id, fc_date_str, relevant_sales_history)

    if "error" in forecast_result:
        return jsonify({"error": forecast_result["error"]}), 400

    expected_qty = forecast_result["prediction"]
    order_qty = math.ceil(expected_qty) if expected_qty > 0 else 0
    product_details = get_product_details(product_id)
    dynamic_price = product_details["base_price"] # Dynamic pricing logic can be added here
    total_co2_kg = estimate_co2(order_qty, product_details["weight_kg"])

    try:
        file_exists = os.path.exists(SUSTAINABILITY_LOG_PATH)
        with open(SUSTAINABILITY_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(SUSTAINABILITY_LOG_PATH).st_size == 0:
                writer.writerow(['date', 'store_id', 'product_id', 'order_qty', 'co2_kg_estimated', 'price_adjusted'])
            writer.writerow([fc_date_str, store_id, product_id, order_qty, round(total_co2_kg, 2), round(dynamic_price, 2)])
    except Exception as e:
        print(f"Warning: Could not write to sustainability.csv: {e}")

    return jsonify({
        "store_id": store_id, "product_id": product_id, "fc_date": fc_date_str,
        "expected_qty": round(expected_qty, 2), "order_qty": order_qty,
        "price": round(dynamic_price, 2), "co2_kg": round(total_co2_kg, 2)
    })

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    # This logic remains the same and is correct.
    try:
        log_df = pd.read_csv(SUSTAINABILITY_LOG_PATH)
        if log_df.empty: return jsonify([]), 200
    except FileNotFoundError:
        return jsonify([]), 200 # Return empty list if no file, frontend will handle it
    
    log_df['date'] = pd.to_datetime(log_df['date'])
    log_df['co2_kg_estimated'] = pd.to_numeric(log_df['co2_kg_estimated'], errors='coerce').fillna(0)
    log_df['month_year'] = log_df['date'].dt.to_period('M')

    monthly_co2_by_store = log_df.groupby(['store_id', 'month_year'])['co2_kg_estimated'].sum().reset_index()
    if monthly_co2_by_store.empty: return jsonify([]), 200

    display_month_period = monthly_co2_by_store['month_year'].max()
    prev_month_period = display_month_period - 1
    
    leaderboard_entries = []
    all_logged_stores = monthly_co2_by_store['store_id'].unique()

    for store_id in all_logged_stores:
        store_monthly_data = monthly_co2_by_store[monthly_co2_by_store['store_id'] == store_id]
        current_month_co2 = store_monthly_data[store_monthly_data['month_year'] == display_month_period]['co2_kg_estimated'].sum()
        prev_month_co2 = store_monthly_data[store_monthly_data['month_year'] == prev_month_period]['co2_kg_estimated'].sum()
        
        if current_month_co2 > 0 or (any(store_monthly_data['month_year'] == display_month_period) and prev_month_co2 > 0):
            badge = "None"
            if current_month_co2 < 100: badge = "Gold"
            elif current_month_co2 < 300: badge = "Silver"
            elif current_month_co2 < 500: badge = "Bronze"

            co2_reduction_percent = 0.0
            if prev_month_co2 > 0:
                co2_reduction_percent = ((prev_month_co2 - current_month_co2) / prev_month_co2) * 100
            elif prev_month_co2 == 0 and current_month_co2 > 0:
                co2_reduction_percent = -100.0

            leaderboard_entries.append({
                "store": store_id, "total_co2_kg": round(current_month_co2, 2), "badge": badge,
                "co2_reduction_percent": round(co2_reduction_percent, 2), "prev_month_co2": round(prev_month_co2, 2)
            })

    sorted_leaderboard = sorted(leaderboard_entries, key=lambda x: x["total_co2_kg"])
    return jsonify(sorted_leaderboard)

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    if not os.path.exists(SUSTAINABILITY_LOG_PATH):
        with open(SUSTAINABILITY_LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['date', 'store_id', 'product_id', 'order_qty', 'co2_kg_estimated', 'price_adjusted'])
    
    # Using host='0.0.0.0' makes it accessible on your local network
    app.run(debug=True, host='0.0.0.0')
