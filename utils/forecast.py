# === inventory_forecast/utils/forecast.py (FINAL CORRECT VERSION) ===
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from datetime import timedelta, date
from sqlalchemy import create_engine, text # Import text for raw SQL queries

# Determine base path for data/models relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODELS_DIR = os.path.join(BASE_DIR, '../models')
DB_FILE = os.path.join(DATA_DIR, 'inventory_small.db') # Path to your SQLite DB

class InventoryForecaster:
    def __init__(self):
        try:
            # Establish database connection
            # check_same_thread=False is needed for SQLite when accessed by multiple threads (Flask requests)
            self.db_engine = create_engine(f'sqlite:///{DB_FILE}', connect_args={'check_same_thread': False})

            # Load model + encoders
            self.model = self._load_model()
            self.encoders = self._load_encoders()

            # Load calendar from DB for general lookup and filtering dates
            self.calendar = self._load_calendar_from_db()

            # Prices are often used for merging, but for dynamic pricing,
            # we'll fetch general product details separately in app.py.
            # No need to load all prices here if they are embedded in sales_long_table.
            # self.prices = self._load_prices_from_db() # This line is commented out as it's not strictly needed for forecast calc if sell_price is in sales_long

            self.features = [
                'event_name_1', 'event_type_1', 'sell_price',
                'lag_7', 'rolling_mean_7', 'day', 'month', 'year', 'weekofyear'
            ]
            print("Forecaster initialized successfully, connected to SQLite.")
        except FileNotFoundError as e:
            print(f"Error loading model/encoder files: {e}. Make sure models/ directory exists and contains .pkl files.")
            raise
        except Exception as e:
            print(f"Error initializing Forecaster or connecting to DB: {e}")
            raise

    def _load_model(self):
        model_path = os.path.join(MODELS_DIR, "inventory_forecast_model.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _load_encoders(self):
        encoder_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
        with open(encoder_path, "rb") as f:
            return pickle.load(f)

    def _load_calendar_from_db(self):
        cal = pd.read_sql_table('calendar', self.db_engine)
        cal['date'] = pd.to_datetime(cal['date'])
        # Ensure 'd' column format matches what sales_long has in DB
        cal['d'] = 'd_' + cal['d'].astype(str).str.replace('d_', '').astype(int).astype(str)
        return cal

    # _load_prices_from_db is not used if sell_price is embedded in sales_long_table
    # def _load_prices_from_db(self):
    #     return pd.read_sql_table('sell_prices', self.db_engine)

    def _get_relevant_sales_history(self, store_id: str, item_id: str, target_forecast_date: date):
        """
        Fetches historical sales data for a specific store-item from the database
        up to the day before the target forecast date.
        This function is crucial for efficient data retrieval.
        """
        # We need enough history for lags (at least 14 days before target_forecast_date)
        # Fetch data up to the day before the forecast
        history_start_date = target_forecast_date - timedelta(days=20) # Get a bit more than needed (e.g., 20 days) for safety
        
        # Use a SQL query to get only the necessary sales history
        query = text(f"""
            SELECT date, sales, sell_price
            FROM sales_long
            WHERE store_id = :store_id
              AND item_id = :item_id
              AND date < :target_date
              AND date >= :history_start_date
            ORDER BY date ASC;
        """)
        
        # Use pandas read_sql to execute query and get DataFrame
        history_df = pd.read_sql(query, self.db_engine,
                                 params={
                                     'store_id': store_id,
                                     'item_id': item_id,
                                     'target_date': target_forecast_date.isoformat(),
                                     'history_start_date': history_start_date.isoformat()
                                 },
                                 parse_dates=['date'])
        
        # Ensure 'date' is datetime type after parsing
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        # Ensure sell_price is filled for the history
        history_df['sell_price'] = history_df['sell_price'].ffill().bfill()
        history_df['sell_price'].fillna(0, inplace=True) # Fallback

        return history_df

    def forecast_quantity(self, store: str, item: str, date_str: str,
                          sales_history_for_item: pd.DataFrame):
        """
        Generates a demand forecast for a given store, item, and date.
        This method now receives the pre-fetched `sales_history_for_item`.
        """
        target_forecast_date = pd.to_datetime(date_str).date() # Ensure it's a date object

        # 1. Get Calendar and Price Info for the Forecast Date
        d_row_cal = self.calendar[self.calendar['date'].dt.date == target_forecast_date]
        if d_row_cal.empty:
            return {"error": f"Invalid forecast date: {date_str} not found in calendar."}

        predict_df = pd.DataFrame([{
            'date': pd.to_datetime(target_forecast_date), # Ensure datetime for dt.day etc.
            'store_id': store,
            'item_id': item,
            'wm_yr_wk': d_row_cal['wm_yr_wk'].iloc[0],
            'event_name_1': d_row_cal['event_name_1'].iloc[0],
            'event_type_1': d_row_cal['event_type_1'].iloc[0],
            'weekday': d_row_cal['weekday'].iloc[0],
        }])

        # Get sell_price for the prediction day from history or by looking up in prices table
        # Since sales_long has sell_price, we can try to get it from the latest history or from prices table.
        # For simplicity, let's just make sure it's consistent for the forecast features
        
        # This part of the code needs to be smart about *future* prices
        # The simplest is to get the latest known price from `sales_history_for_item`
        if not sales_history_for_item.empty and 'sell_price' in sales_history_for_item.columns:
            predict_df['sell_price'] = sales_history_for_item['sell_price'].iloc[-1] # Get latest known price
        else:
            # Fallback: get from the prices table if needed, or default
            # This requires prices_df to be loaded in Forecaster init.
            # For this exact setup, assume it comes from sales_long_history
            print("Warning: No sell_price found in sales_history_for_item. Defaulting to 0 for feature.")
            predict_df['sell_price'] = 0.0 # Default if no price history

        predict_df['sell_price'] = predict_df['sell_price'].ffill().bfill()
        if predict_df['sell_price'].isnull().any(): # Final check after fills
            predict_df['sell_price'].fillna(0, inplace=True) # Fallback

        for col in ['event_name_1', 'event_type_1', 'weekday']:
            if col in predict_df.columns:
                val = predict_df[col].iloc[0]
                if val in self.encoders[col].classes_:
                    predict_df[col] = self.encoders[col].transform([val])[0]
                else:
                    predict_df[col] = -1

        sales_history_for_item.sort_values(by='date', inplace=True)

        lag_7_date = target_forecast_date - timedelta(days=7)
        lag_7_value_row = sales_history_for_item[sales_history_for_item['date'].dt.date == lag_7_date]['sales']
        predict_df['lag_7'] = lag_7_value_row.iloc[0] if not lag_7_value_row.empty else 0

        rolling_mean_end_date = target_forecast_date - timedelta(days=8)
        rolling_mean_start_date = rolling_mean_end_date - timedelta(days=6)

        rolling_data = sales_history_for_item[
            (sales_history_for_item['date'].dt.date >= rolling_mean_start_date) &
            (sales_history_for_item['date'].dt.date <= rolling_mean_end_date)
        ]['sales']

        predict_df['rolling_mean_7'] = rolling_data.mean() if not rolling_data.empty else 0

        predict_df['day'] = predict_df['date'].dt.day
        predict_df['month'] = predict_df['date'].dt.month
        predict_df['year'] = predict_df['date'].dt.year
        predict_df['weekofyear'] = predict_df['date'].dt.isocalendar().week.astype(int)

        # 4. Make Prediction
        try:
            for f in self.features:
                if f not in predict_df.columns:
                    predict_df[f] = 0
            
            X_pred = predict_df[self.features]

            pred = self.model.predict(X_pred)[0]
            current_sell_price_feature_val = predict_df['sell_price'].values[0] # Price used as feature for model

            return {
                "prediction": round(pred, 2),
                "price": round(current_sell_price_feature_val, 2) # This is the price used as a feature, not necessarily final display price
            }
        except KeyError as e:
            return {"error": f"Missing feature for prediction: {e}. Available columns: {predict_df.columns.tolist()}"}
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
