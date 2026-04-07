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
def get_historical_data():
    """Load and cache the full merged dataset."""
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
    for _ in range(iterations):
        # calling model(X, training=True) keeps dropout layers active
        preds_scaled = model(X, training=True)
        
        # Inverse transform only the relevant closing price
        # (Need to reshape for the scaler)
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
    
    for i in range(len(full_df) - depth, len(full_df)):
        # Data window: [i-60, i]
        window = full_df.iloc[i - lookback : i]
        scaled_window = scaler.transform(window.values)
        X = np.expand_dims(scaled_window, axis=0)
        # 1-day ahead is the first element of the forecast
        pred_scaled = model(X, training=False)[0, 0] 
        
        num_features = scaler.n_features_in_
        dummy = np.zeros((1, num_features))
        dummy[0, 3] = pred_scaled
        pred_unscaled = scaler.inverse_transform(dummy)[0, 3]
        historical_preds.append(pred_unscaled)
        
    return pd.Series(historical_preds, index=backtest_dates)

def run_sentiment_drift_analysis(model, scaler, target_month_data):
    # Simplified placeholder for drift logic
    actual_sentiment = target_month_data['Sentiment'].mean()
    drift_score = np.random.uniform(-5, 5)
    return actual_sentiment + drift_score, actual_sentiment - drift_score

# Main UI
st.title("BTC Profit and Sentiment Predictor")
st.markdown("### Powered by LSTM and Market Psychology")

model, scaler = load_assets()
full_df = get_historical_data()

if model and scaler and not full_df.empty:
    # Sidebar for Simulation Control
    with st.sidebar:
        st.header("Simulation Settings")
        selected_date = st.date_input("Investment Date", value=full_df.index[-1].date())
        
        idx = full_df.index.get_indexer([pd.to_datetime(selected_date)], method='nearest')[0]
        auto_price = float(full_df.iloc[idx]['Close'])
        
        entry_price = st.number_input("Entry Price (USD)", value=auto_price)
        investment = st.number_input("Investment Amount ($)", value=1000)
        profit_target = st.number_input("Profit Target (%)", value=2.0, step=0.5)
        
        if st.button("Run Simulation", use_container_width=True):
            with st.spinner("Calculating probability bands..."):
                recent_data = full_df.tail(cloud_config.LOOKBACK_DAYS)
                mean, std = predict_with_uncertainty(model, scaler, recent_data)
                
                # Use PREVIOUSLY SAVED Calibration (Instant)
                state = lifecycle.load_calibration_state()
                avg_drift = state['drift_value']
                
                # Applied Correction (Future Forecast)
                correction_impact = avg_drift * 500 
                calibrated_prices = mean + correction_impact
                
                forecast_start = full_df.index[-1] + timedelta(days=1)
                forecast_dates = [forecast_start + timedelta(days=i) for i in range(cloud_config.FORECAST_DAYS)]
                backtest_series = get_backtest_predictions(model, scaler, full_df)
                
                st.session_state['results'] = {
                    'dates': forecast_dates,
                    'prices': mean,
                    'calibrated_prices': calibrated_prices,
                    'std': std,
                    'target': entry_price * (1 + profit_target/100),
                    'target_pct': profit_target,
                    'backtest': backtest_series,
                    'avg_drift': avg_drift,
                    'entry_price': entry_price,
                    'investment': investment
                }
        
        if 'results' in st.session_state:
            res = st.session_state['results']
            if st.button("Save this Simulation", use_container_width=True):
                inv_mgr.save_investment(
                    amount=res['investment'],
                    date=selected_date,
                    price=res['entry_price'],
                    forecast_prices=res['prices'],
                    calibrated_prices=res['calibrated_prices'],
                    std=res['std'],
                    forecast_dates=res['dates'],
                    profit_target=res['target_pct']
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

            # Retrain Signaling
            needs_retrain = (age > 30) or (abs(drift_val) > 15) or (error_pct > 10)
            
            if needs_retrain:
                st.error("⚠️ RETRAIN RECOMMENDED")
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
                st.success("✅ Model is Healthy")

            st.divider()
            if st.button("Recalculate Calibration"):
                with st.spinner("Running 30-day Batch Analysis..."):
                    drift_df = calib.batch_calibrate_sentiment(model, scaler, full_df)
                    new_drift = drift_df['Drift'].tail(30).mean()
                    latest_price = full_df['Close'].iloc[-1]
                    lifecycle.save_calibration_state(new_drift, latest_price)
                    
                    # Store chart in session for viewing
                    drift_fig = go.Figure()
                    drift_fig.add_trace(go.Scatter(x=drift_df.index, y=drift_df['Actual_Sentiment'], name='Actual', line=dict(color='gray', dash='dot')))
                    drift_fig.add_trace(go.Scatter(x=drift_df.index, y=drift_df['Market_Aligned'], name='Aligned', line=dict(color='cyan', width=2)))
                    drift_fig.update_layout(height=200, template="plotly_dark", paper_bgcolor='black', plot_bgcolor='black', margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                    st.session_state['health_chart'] = drift_fig
                    
                    st.success("Calibration complete!")
                    st.rerun()

            if 'health_chart' in st.session_state:
                st.plotly_chart(st.session_state['health_chart'], use_container_width=True)
                if st.button("Clear Analysis"):
                    del st.session_state['health_chart']
                    st.rerun()

    # Main Tabs
    tab1, tab2 = st.tabs(["Market Analysis", "Investment Journal"])
    
    with tab1:
        curr_price = full_df['Close'].iloc[-1]
        cols = st.columns(3)
        cols[0].metric("Current BTC", f"${curr_price:,.2f}")
        
        if 'results' in st.session_state:
            res_entry = st.session_state['results']['entry_price']
            roi = ((curr_price / res_entry) - 1) * 100
            cols[1].metric("Simulation Entry", f"${res_entry:,.2f}")
            cols[2].metric("Current ROI", f"{roi:.2f}%")

        # Plotly Chart
        fig = go.Figure()
        
        # 1. Historical Actual Price (Last 90 days)
        hist_view = full_df.tail(90)
        fig.add_trace(go.Scatter(
            x=hist_view.index, y=hist_view['Close'],
            name='Actual Price', line=dict(color='white', width=2)
        ))
        
        if 'results' in st.session_state:
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
            
            # 3b. Calibrated Forecast (V3)
            fig.add_trace(go.Scatter(
                x=res['dates'], y=res['calibrated_prices'],
                name='Sentiment-Calibrated', 
                line=dict(color='#ffaa00', width=2, dash='dash')
            ))
            
            # 4. Target Line
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
                roi = ((curr_price / inv['price']) - 1) * 100
                with st.expander(f"{inv['date']} | ROI: {roi:.2f}% | Entry: ${inv['price']:,.0f}"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.write(f"**Amount:** ${inv['amount']}")
                        st.write(f"**Target:** {inv.get('profit_target', 2.0)}%")
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
