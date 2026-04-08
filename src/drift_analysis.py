import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import joblib
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import cloud_config as cloud_config
from data_loader import prepare_merged_dataset, create_sequences
import investments_manager as inv_mgr
import calibration as calib
import model_lifecycle as lifecycle
import prediction_logger as pred_log

def calculate_withdrawal_date(dates, prices, target_price):
    """Find the first date where price meets or exceeds target."""
    for i, p in enumerate(prices):
        if p >= target_price:
            return dates[i]
    return None

from google.cloud import storage

st.set_page_config(page_title="BTC Predictor Dashboard", layout="wide")

# Helper to load assets (Local or GCS Sync)
@st.cache_resource
def load_assets():
    # Ensure Model Directory exists
    os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)

    # 1. Sync from GCS if local is missing (Cloud Run Cold Start)
    if not os.path.exists(cloud_config.MODEL_PATH) or not os.path.exists(cloud_config.SCALER_PATH):
        with st.status("Syncing model from Google Cloud Storage...", expanded=True) as status:
            try:
                client = storage.Client()
                bucket = client.bucket(cloud_config.BUCKET_NAME)
                
                # Download Model
                if not os.path.exists(cloud_config.MODEL_PATH):
                    st.write("Downloading model...")
                    blob = bucket.blob(f"{cloud_config.MODEL_DIR}/btc_lstm_model.h5")
                    blob.download_to_filename(cloud_config.MODEL_PATH)
                
                # Download Scaler
                if not os.path.exists(cloud_config.SCALER_PATH):
                    st.write("Downloading scaler...")
                    blob = bucket.blob(f"{cloud_config.MODEL_DIR}/scaler.pkl")
                    blob.download_to_filename(cloud_config.SCALER_PATH)
                
                status.update(label="Sync Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Sync Failed!", state="error", expanded=True)
                raise RuntimeError(f"Could not retrieve model from GCS: {str(e)}")
    
    # 2. Final Load
    try:
        model = keras.models.load_model(cloud_config.MODEL_PATH, compile=False)
        scaler = joblib.load(cloud_config.SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

@st.cache_data
def get_full_dataset():
    """Load and cache the full and cleaned merged datasets."""
    return prepare_merged_dataset()

def predict_with_uncertainty(model, scaler, current_data, iterations=20):
    """
    Predict next 30 days using Monte Carlo Dropout for 68% CI (1-Sigma).
    """
    scaled_input = scaler.transform(current_data.values)
    X = np.expand_dims(scaled_input, axis=0)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    # Perform multiple passes with dropout active
    preds_list = []
    
    # Keras 3 Stability: Wrap the functional call to maintain name_scope_stack
    @tf.function(reduce_retracing=True)
    def mc_step(X_batch):
        return model(X_batch, training=True)

    for _ in range(iterations):
        preds_scaled = mc_step(X)
        
        # Inverse transform only the relevant closing price
        num_features = scaler.n_features_in_
        dummy = np.zeros((cloud_config.FORECAST_DAYS, num_features))
        dummy[:, 3] = preds_scaled[0]
        preds_unscaled = scaler.inverse_transform(dummy)[:, 3]
        preds_list.append(preds_unscaled)
        
    preds_list = np.array(preds_list)
    mean = preds_list.mean(axis=0)
    std = preds_list.std(axis=0) # 1-Sigma (68% CI)
    
    return mean, std

def get_backtest_predictions(model, scaler, full_df, depth=30):
    """
    Generate 1-day-ahead predictions for the last 'depth' days.
    Ensures NO data leakage.
    """
    lookback = cloud_config.LOOKBACK_DAYS
    backtest_dates = full_df.index[-depth:]
    historical_preds = []
    
    # Keras 3 Stability: Define a static call
    @tf.function(reduce_retracing=True)
    def call_model(X_in):
        return model(X_in, training=False)

    for i in range(len(full_df) - depth, len(full_df)):
        # Data window: [i-60, i]
        window = full_df.iloc[i - lookback : i]
        scaled_window = scaler.transform(window.values)
        X = np.expand_dims(scaled_window, axis=0)
        # 1-day ahead is the first element of the forecast
        pred_scaled = call_model(X)[0, 0] 
        
        num_features = scaler.n_features_in_
        dummy = np.zeros((1, num_features))
        dummy[0, 3] = pred_scaled
        pred_unscaled = scaler.inverse_transform(dummy)[0, 3]
        historical_preds.append(pred_unscaled)
        
    return pd.Series(historical_preds, index=backtest_dates)

# Main UI
st.title("BTC Profit and Sentiment Predictor")
st.markdown("### Powered by LSTM and Market Psychology")

model, scaler = load_assets()
full_df, clean_df = get_full_dataset()

if model and scaler and not full_df.empty:
    # Sidebar for Simulation Control
    with st.sidebar:
        st.header("Simulation Settings")
        selected_date = st.date_input("Investment Date", value=clean_df.index[-1].date())
        
        idx = clean_df.index.get_indexer([pd.to_datetime(selected_date)], method='nearest')[0]
        auto_price = float(clean_df.iloc[idx]['Close'])
        
        entry_price = st.number_input("Entry Price (USD)", value=auto_price)
        investment = st.number_input("Investment Amount ($)", value=1000)
        profit_target = st.number_input("Profit Target (%)", value=2.0, step=0.5)
        
        def generate_base_forecast():
            """Standardized model forecasting (Runs once at startup)."""
            
            # 1. AUTOMATED DRIFT UPDATE (Run FIRST to calibrate)
            with st.spinner("Analyzing RSS Sentiment & Market Alignment..."):
                drift_df = calib.batch_calibrate_sentiment(model, scaler, clean_df, depth=3) 
                avg_drift = drift_df['Drift'].mean()
                latest_price = clean_df['Close'].iloc[-1]
                lifecycle.save_calibration_state(avg_drift, latest_price)

            # 2. APPLY DRIFT TO INPUT DATA
            # We copy the tail and shift the Sentiment column to align with market weight
            recent_data = clean_df.tail(cloud_config.LOOKBACK_DAYS).copy()
            
            # Use the drift calculated in calibration.py
            recent_data['Sentiment'] = np.clip(recent_data['Sentiment'] + avg_drift, 0, 100)
            
            # 3. FORECAST ON CALIBRATED DATA
            mean, std = predict_with_uncertainty(model, scaler, recent_data, iterations=50)
            tight_std = std * 0.5
            
            # Use CLEAN data tail for forecast start
            forecast_start = clean_df.index[-1] + timedelta(days=1)
            forecast_dates = [forecast_start + timedelta(days=i) for i in range(cloud_config.FORECAST_DAYS)]
            backtest_series = get_backtest_predictions(model, scaler, clean_df)
            
            # Persist this heartbeat (log to daily history)
            pred_log.log_predictions(forecast_dates, mean)
            
            return {
                'dates': forecast_dates,
                'prices': mean,
                'std': tight_std,
                'backtest': backtest_series,
                'avg_drift': avg_drift
            }

        def calculate_withdrawal_plan(base_res, entry_price, profit_target, investment):
            """Instant calculation for investment strategy on top of existing forecast."""
            target_val = entry_price * (1 + profit_target/100)
            withdrawal_date = calculate_withdrawal_date(base_res['dates'], base_res['prices'], target_val)
            
            return {
                **base_res,
                'target': target_val,
                'target_pct': profit_target,
                'entry_price': entry_price,
                'investment': investment,
                'withdrawal_date': withdrawal_date
            }

        # Auto-run base forecast on first load
        if 'base_forecast' not in st.session_state:
            with st.spinner("Initializing Latest Market Forecast..."):
                st.session_state['base_forecast'] = generate_base_forecast()
        
        # Initial Results from the base forecast (Quietly setup)
        if 'results' not in st.session_state:
            st.session_state['results'] = calculate_withdrawal_plan(
                st.session_state['base_forecast'], auto_price, profit_target, investment
            )
            st.session_state['plan_triggered'] = False

        if st.button("Run Simulation", use_container_width=True):
            # Instant calculation - no spinner needed for the model anymore
            st.session_state['results'] = calculate_withdrawal_plan(
                st.session_state['base_forecast'], entry_price, profit_target, investment
            )
            st.session_state['plan_triggered'] = True
            st.success("Withdrawal plan updated!")
        
        if 'results' in st.session_state:
            res = st.session_state['results']
            if st.button("Save this Simulation", use_container_width=True):
                inv_mgr.save_investment(
                    amount=res['investment'],
                    date=selected_date,
                    price=res['entry_price'],
                    forecast_prices=res['prices'],
                    calibrated_prices=res['prices'], # Mean as baseline
                    std=res['std'],
                    forecast_dates=res['dates'],
                    profit_target=res['target_pct'],
                    original_withdrawal_date=res['withdrawal_date']
                )
                st.success("Investment saved to Journal")
                st.rerun()

        st.divider()
        with st.expander("System Health & Calibration"):
            age, _ = lifecycle.get_model_info()
            st.write(f"**Model Age:** {age} days")
            
            cal_state = lifecycle.load_calibration_state()
            drift_val = cal_state['drift_value']
            st.write(f"**Status:** {cal_state['status']}")
            st.write(f"**Active Drift:** {drift_val:+.2f} pts")
            
            # Calculate Recent Accuracy (RMSE Proxy)
            error_pct = 0.0
            if 'results' in st.session_state:
                res = st.session_state['results']
                backtest = res['backtest']
                actuals = full_df.loc[backtest.index, 'Close']
                error_pct = (np.abs(backtest - actuals) / actuals).mean() * 100
                st.write(f"**Recent Error (MAE):** {error_pct:.2f}%")
            else:
                st.write("**Recent Error:** Run Simulation to evaluate")
            
            st.divider()
            if st.button("Recalibrate Market Alignment", use_container_width=True):
                for key in ['base_forecast', 'results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.info("Recalibrating psychological drift...")
                st.rerun()

            if st.button("Force Market Refresh", use_container_width=True):
                st.cache_data.clear()
                for key in ['base_forecast', 'results', 'plan_triggered']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Cache cleared. Re-fetching data...")
                st.rerun()
            
            # Retrain Signaling
            needs_retrain = (age > 30) or (abs(drift_val) > 15) or (error_pct > 10)
            
            if needs_retrain:
                st.error("RETRAIN RECOMMENDED")
                if age > 30: st.caption("- Model is older than 30 days")
                if abs(drift_val) > 15: st.caption("- Sentiment Drift exceeds 15pts")
                if error_pct > 10: st.caption("- Prediction Error exceeds 10%")
                
                if st.button("Prepare GCP Training Job"):
                    with st.spinner("Submitting Spot Training Job to Vertex AI..."):
                        try:
                            image_uri = f"gcr.io/{cloud_config.PROJECT_ID}/btc-trainer"
                            import train_on_gcp
                            job = train_on_gcp.launch_vertex_ai_job(image_uri)
                            st.success(f"Job Submitted! ID: {job.resource_name}")
                            st.info("Monitor progress in your GCP Console (Vertex AI > Training)")
                        except Exception as e:
                            st.error(f"Failed to submit: {str(e)}")
                            st.write("Ensure you are logged into gcloud locally.")
            else:
                st.success("Model is Healthy")

    # Main Tabs
    tab1, tab2 = st.tabs(["Market Analysis", "Investment Journal"])
    
    with tab1:
        # LIVE PRICE from undropped full_df
        latest_price_val = full_df['Close'].iloc[-1]
        latest_date_val = full_df.index[-1]
        
        # MODEL FORECAST for Today
        # Use prices[0] and dates[0] from the base forecast (it starts today)
        forecast_today_val = None
        forecast_today_date = None
        if 'base_forecast' in st.session_state:
            forecast_today_val = float(st.session_state['base_forecast']['prices'][0])
            forecast_today_date = st.session_state['base_forecast']['dates'][0].strftime('%Y-%m-%d')
        
        st.subheader("Market Summary")
        mcols = st.columns(3)
        mcols[0].metric(f"Live BTC ({latest_date_val.strftime('%H:%M')})", f"${latest_price_val:,.2f}")
        
        if forecast_today_val:
            diff = latest_price_val - forecast_today_val
            diff_pct = (diff / forecast_today_val) * 100
            mcols[1].metric(f"Forecast ({forecast_today_date})", f"${forecast_today_val:,.2f}")
            mcols[2].metric("Difference", f"${diff:,.2f}", f"{diff_pct:+.2f}% vs Forecast")
        else:
            mcols[1].write("**Forecast:** Loading...")
            mcols[2].write("**Difference:** Pending")
        
        # Only show simulation results if the user has explicitly triggered a plan
        if st.session_state.get('plan_triggered') and st.session_state['results'].get('withdrawal_date'):
            w_date = st.session_state['results']['withdrawal_date']
            st.success(f"Predicted Profit Target Hit Date: {w_date.strftime('%Y-%m-%d')}")
        elif st.session_state.get('plan_triggered'):
            st.warning("Profit target not reached within the 30-day forecast window.")

        # Plotly Chart
        fig = go.Figure()
        
        # 1. Historical Actual Price (Last 90 days)
        hist_view = full_df.tail(90)
        fig.add_trace(go.Scatter(
            x=hist_view.index, y=hist_view['Close'],
            name='Actual Price', line=dict(color='white', width=2)
        ))
        
        # Dual Performance Summaries
        if 'results' in st.session_state:
            st.divider()
            
            history_df = pred_log.get_performance_stats()
            
            if not history_df.empty:
                # 1. Historical Performance (Yesterday D-1)
                # Yesterday is clean_df.index[-1]
                yesterday_date = clean_df.index[-1].strftime('%Y-%m-%d')
                actual_yesterday = clean_df['Close'].iloc[-1]
                
                # We want predictions made ON 'Yesterday-1' FOR 'Yesterday'
                day_before_yesterday = (clean_df.index[-1] - timedelta(days=1)).strftime('%Y-%m-%d')
                pred_yesterday = history_df[
                    (history_df['sim_run_date'] == day_before_yesterday) & 
                    (history_df['forecast_date'] == yesterday_date)
                ]['predicted_price']
                
                # 2. Live Today Snapshot (D)
                # Predictions made TODAY for TODAY
                today_tag = datetime.now().strftime('%Y-%m-%d')
                pred_today = history_df[
                    (history_df['sim_run_date'] == today_tag) & 
                    (history_df['forecast_date'] == today_tag)
                ]['predicted_price']
                
                st.subheader("Model Performance Summary")
                mcols = st.columns(2)
                
                with mcols[0]:
                    st.write("**Yesterday's Accuracy (D-1)**")
                    if not pred_yesterday.empty:
                        p_mean = pred_yesterday.mean()
                        st.metric(f"Market Close ({yesterday_date})", f"${actual_yesterday:,.2f}")
                        st.metric("Predicted Mean", f"${p_mean:,.2f}", delta=f"{((p_mean/actual_yesterday)-1)*100:.2f}% Error")
                    else:
                        st.info("No predictions found from Day D-2 for yesterday.")
                
                with mcols[1]:
                    st.write("**Live Today Snapshot (D)**")
                    if not pred_today.empty:
                        p_today_mean = pred_today.mean()
                        run_count = len(pred_today)
                        run_list = ", ".join([f"${v:,.0f}" for v in pred_today.tolist()])
                        
                        st.metric(
                            "Session Predicted Mean", 
                            f"${p_today_mean:,.2f}", 
                            delta=f"{((p_today_mean/latest_price_val)-1)*100:.2f}% vs Live",
                            help=f"Aggregated from {run_count} simulations today.\nIndividual values: {run_list}"
                        )
                    else:
                        st.info("Run a simulation today to populate live metrics.")
            
            st.divider()
            res = st.session_state['results']
            
            # 2. Historical Model Suggestions (Backtest)
            fig.add_trace(go.Scatter(
                x=res['backtest'].index, y=res['backtest'].values,
                name='Model (1D Backtest)', 
                line=dict(color='cyan', dash='dot', width=1)
            ))
            
            # 3. Forecast with 68% CI
            upper_bound = res['prices'] + res['std']
            lower_bound = res['prices'] - res['std']
            
            fig.add_trace(go.Scatter(
                x=res['dates'], y=upper_bound,
                mode='lines', name='Upper Bound',
                line=dict(color='rgba(0, 255, 0, 0.2)', width=1, dash='dot'),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=res['dates'], y=lower_bound,
                mode='lines', name='Lower Bound',
                line=dict(color='rgba(0, 255, 0, 0.2)', width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)',
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=res['dates'], y=res['prices'],
                name='Forecast Mean', line=dict(color='#00ff00', width=3)
            ))
            
            # 4. Target Line (Only if triggered)
            if st.session_state.get('plan_triggered'):
                fig.add_trace(go.Scatter(
                    x=res['dates'], y=[res['target']]*len(res['dates']),
                    name=f"{res['target_pct']}% Profit Target", line=dict(color='red', dash='dash')
                ))
            
        fig.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor='black',
            plot_bgcolor='black',
            hovermode='x unified',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=True, gridcolor='#222'),
            yaxis=dict(showgrid=True, gridcolor='#222')
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Saved Investments")
        investments = inv_mgr.load_investments()
        if not investments:
            st.info("No saved investments. Run a simulation and click 'Save this Simulation' to start journaling.")
        else:
            for inv in reversed(investments):
                roi = ((latest_price_val / inv['price']) - 1) * 100
                with st.expander(f"{inv['date']} | ROI: {roi:.2f}% | Entry: ${inv['price']:,.0f}"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.write(f"**Amount:** ${inv['amount']}")
                        st.write(f"**Target:** {inv.get('profit_target', 2.0)}%")
                        
                        # Comparison of Target Dates
                        orig_date_str = inv.get('original_withdrawal_date', 'N/A')
                        base_res = st.session_state.get('base_forecast', {})
                        curr_price_forecast = base_res.get('prices', [])
                        curr_dates = base_res.get('dates', [])
                        
                        target_price = inv['price'] * (1 + inv.get('profit_target', 2.0)/100)
                        curr_date = calculate_withdrawal_date(curr_dates, curr_price_forecast, target_price)
                        
                        st.write(f"**Original Target Date:** {orig_date_str[:10]}")
                        if curr_date:
                            curr_date_str = curr_date.strftime('%Y-%m-%d')
                            st.write(f"**Current Target Date:** {curr_date_str}")
                            
                            if orig_date_str != 'None' and orig_date_str != 'N/A':
                                try:
                                    od = datetime.strptime(orig_date_str[:10], '%Y-%m-%d')
                                    diff = (curr_date - od).days
                                    if diff < 0:
                                        st.success(f"Moving Closer! ({abs(diff)} days earlier)")
                                    elif diff > 0:
                                        st.warning(f"Receding... ({diff} days later)")
                                    else:
                                        st.info("On Track (No change)")
                                except:
                                    pass
                        else:
                            st.error("Target no longer in 30-day window")

                        if st.button("Delete Record", key=inv['id']):
                            inv_mgr.delete_investment(inv['id'])
                            st.rerun()
                    
                    # Mini Chart with Historical Overlay
                    if "forecast_prices" in inv and inv["forecast_prices"]:
                        f_dates = [pd.to_datetime(d) for d in inv['forecast_dates']]
                        f_prices = np.array(inv['forecast_prices'])
                        f_std = np.array(inv['std'])
                        f_calib = np.array(inv['calibrated_prices'])
                        target_price = inv['price'] * (1 + inv.get('profit_target', 2.0)/100)
                        
                        upper_bound = f_prices + f_std
                        lower_bound = f_prices - f_std
                        
                        m_fig = go.Figure()
                        
                        # Actual Prices that happened since the investment
                        actual_since = full_df[full_df.index >= f_dates[0]]
                        if not actual_since.empty:
                            m_fig.add_trace(go.Scatter(x=actual_since.index, y=actual_since['Close'], name='Actual', line=dict(color='white', width=2)))
                        
                        # 68% Confidence Interval (Snapshot)
                        m_fig.add_trace(go.Scatter(x=f_dates, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        m_fig.add_trace(go.Scatter(
                            x=f_dates, y=lower_bound,
                            mode='lines', line=dict(width=0),
                            fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)',
                            name='68% CI'
                        ))

                        # Original Forecast
                        m_fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='Model', line=dict(color='#00ff00', width=1)))
                        m_fig.add_trace(go.Scatter(x=f_dates, y=f_calib, name='Calibrated', line=dict(color='#ffaa00', width=1, dash='dash')))
                        
                        # Profit Target Line
                        m_fig.add_trace(go.Scatter(x=f_dates, y=[target_price]*len(f_dates), name='Target', line=dict(color='red', width=1, dash='dot')))
                        
                        # Lower-Bound Profit Marker (Where Lower CI crosses Target)
                        lower_bound = f_prices - f_std
                        profit_day = -1
                        for i, lb in enumerate(lower_bound):
                            if lb >= target_price:
                                profit_day = i
                                break
                        
                        if profit_day != -1:
                            m_fig.add_trace(go.Scatter(
                                x=[f_dates[profit_day]], y=[lower_bound[profit_day]],
                                mode='markers', name='Safe Profit Point',
                                marker=dict(color='yellow', size=10, symbol='diamond')
                            ))
                        
                        m_fig.update_layout(height=250, template="plotly_dark", paper_bgcolor='black', plot_bgcolor='black', margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                        st.plotly_chart(m_fig, use_container_width=True)

    st.divider()
    st.subheader("Model Status")
    
    # 1. High Drift Detection (using cached state)
    cal_state = lifecycle.load_calibration_state()
    drift_val = cal_state['drift_value']
    
    if abs(drift_val) > 10:
        st.error(f"HIGH DRIFT: {drift_val:+.1f} points. Market psychology is decoupled from price action.")
    elif abs(drift_val) > 5:
        st.warning(f"MODERATE DRIFT: {drift_val:+.1f} points. Manual calibration refresh recommended.")
    else:
        st.success(f"STABLE ALIGNMENT: {drift_val:+.1f} points. Model behavior is synced with market sentiment.")
    
    st.info("System is optimized using the latest stored calibration. Recalibrate in the sidebar if market conditions change significantly.")
