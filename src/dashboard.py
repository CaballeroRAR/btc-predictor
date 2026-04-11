import streamlit as st
st.set_page_config(layout="wide", page_title="BTC Pulse Predictor", page_icon="📈")
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv

# 1. Configuration & Secrets
load_dotenv()
import cloud_config
from data_loader import prepare_merged_dataset
import investments_manager as inv_mgr
import model_lifecycle as lifecycle
import prediction_logger as pred_log
import ui_blocks as ui
from database import DatabaseManager

# 2. Logic Service
import dashboard_logic as logic

# --- SECURITY GATEWAY ---
def check_password():
    """Returns True if the user had the correct password."""
    if st.session_state.get("authenticated"):
        return True

    PASSWORD = os.getenv("DASHBOARD_PASSWORD", "bitcoin2024")

    # Use centered columns for the login box
    _, center_col, _ = st.columns([1, 1, 1])

    with center_col:
        st.title("BTC Predictor Tool")
        with st.form("login"):
            password_input = st.text_input("Enter Dashboard Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                if password_input == PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        
        st.caption("Contact: caballero.data.scientist@tutamail.com for access credentials.")
    return False

# Execution Guard
if not check_password():
    st.stop()

# --- INITIALIZATION ---
db_mgr = DatabaseManager()
model, scaler = lifecycle.load_assets()

# Sidebar: Force Refresh
st.sidebar.header("Global Controls")
force_refresh = st.sidebar.button("Force Market Refresh", use_container_width=True)

if force_refresh:
    st.cache_data.clear()
    st.session_state.clear()
    st.session_state["authenticated"] = True
    st.rerun()

@st.cache_data
def load_market_data(force=False):
    return prepare_merged_dataset(force_refresh=force)

full_df, clean_df = load_market_data(force=force_refresh)

# UI Layout
st.title("BTC Profit and Sentiment Predictor")
st.markdown("### Powered by LSTM and Market Psychology")

if model and scaler and not full_df.empty:
    # 3. Sync Actuals (Non-blocking spinner)
    if 'actuals_updated' not in st.session_state:
        with st.spinner("Syncing latest market evaluation..."):
            pred_log.update_actuals(full_df=full_df)
            st.session_state['actuals_updated'] = True

    # 4. Orchestrate Forecast
    if 'base_forecast' not in st.session_state:
        # Business logic is decoupled; only UI feedback is here
        with st.spinner("Initializing Latest Market Forecast..."):
            st.session_state['base_forecast'] = logic.get_base_forecast(db_mgr, model, scaler, clean_df)
    
    base_res = st.session_state['base_forecast']
    if base_res.get('is_cached'):
        # Fallback to 'timestamp' if calculation_time is missing (for older cache compatibility)
        ts = base_res.get('calculation_time') or base_res.get('timestamp')
        if ts:
            st.caption(f"Loaded from cache (Generated {ts.strftime('%H:%M')} UTC)")

    # 5. Sidebar Simulation Controls
    with st.sidebar:
        st.divider()
        st.header("Simulation Settings")
        selected_date = st.date_input("Investment Date", value=clean_df.index[-1].date())
        
        idx = clean_df.index.get_indexer([pd.to_datetime(selected_date)], method='nearest')[0]
        auto_price = float(clean_df.iloc[idx]['Close'])
        entry_price = st.number_input("Entry Price (USD)", value=auto_price)
        investment = st.number_input("Investment Amount ($)", value=1000)
        profit_target = st.number_input("Profit Target (%)", value=2.0, step=0.5)
        
        # Immediate calculation based on stored base forecast
        if 'results' not in st.session_state:
            st.session_state['results'] = logic.calculate_withdrawal_plan(base_res, auto_price, profit_target, investment)
            st.session_state['plan_triggered'] = False

        if st.button("Run Simulation", use_container_width=True):
            st.session_state['results'] = logic.calculate_withdrawal_plan(base_res, entry_price, profit_target, investment)
            st.session_state['plan_triggered'] = True
        
        if st.button("Save this Simulation", use_container_width=True):
            r = st.session_state['results']
            inv_mgr.save_investment(
                amount=r['investment'], date=selected_date, price=r['entry_price'], 
                forecast_prices=r.get('raw_prices', r['prices']), 
                calibrated_prices=r['prices'], 
                std=r['std'], 
                forecast_dates=r['dates'], 
                profit_target=r['target_pct'], 
                original_withdrawal_date=r['withdrawal_date']
            )
            st.success("Investment saved!")

        st.divider()
        with st.expander("System Health"):
            age, last_trained = lifecycle.get_model_info()
            st.write(f"**Model Age:** {age} days")
            if last_trained:
                st.write(f"**Model Updated:** {last_trained.strftime('%Y-%m-%d')}")
            
            # Display Last Prediction Time
            last_run = base_res.get('calculation_time') or base_res.get('timestamp')
            if last_run:
                st.write(f"**Last Prediction:** {last_run.strftime('%H:%M:%S UTC')}")
            
            # Use the drift directly from our results
            current_drift = base_res.get('avg_drift', 0.0)
            st.write(f"**Drift:** {current_drift:+.2f} pts")
            
            if st.button("Recalibrate Market Alignment", use_container_width=True):
                st.session_state['base_forecast'] = logic.get_base_forecast(db_mgr, model, scaler, clean_df, force=True)
                st.rerun()

        st.divider()
        with st.expander("Model Management"):
            import vertex_trigger as vertex
            
            # 1. Job Telemetry (Active Monitoring)
            st.markdown("### Training Status")
            active_jobs = vertex.get_latest_training_jobs(limit=1)
            if active_jobs:
                job = active_jobs[0]
                status_txt = vertex.get_status_summary(job)
                st.info(status_txt)
                if st.button("Refresh Job Status", use_container_width=True):
                    st.rerun()
            else:
                st.write("No active training jobs found.")

            st.divider()
            # 2. Training Trigger
            st.markdown("### Retraining")
            st.warning("Launching a new training job will provision a GCP compute instance. This will incur cloud costs.")
            
            confirm_training = st.checkbox("Confirm Cloud Training Trigger")
            if st.button("Launch Vertex AI Model Training", disabled=not confirm_training, use_container_width=True):
                with st.spinner("Provisioning Vertex AI CustomJob..."):
                    try:
                        new_job = vertex.trigger_training_job()
                        st.success(f"Job successfully created: {new_job.display_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to launch job: {str(e)}")

    # 6. Main Visualizations
    tab1, tab2 = st.tabs(["Market Analysis", "Investment Journal"])
    
    with tab1:
        latest_price_val = full_df['Close'].iloc[-1]
        latest_date_val = full_df.index[-1]
        res = st.session_state['base_forecast']
        
        ui.render_market_summary_metrics(latest_price_val, latest_date_val, float(res['prices'][0]), res['dates'][0].strftime('%Y-%m-%d'))
        
        if st.session_state.get('plan_triggered') and st.session_state['results'].get('withdrawal_date'):
            st.success(f"Predicted Profit Target Hit Date: {st.session_state['results']['withdrawal_date'].strftime('%Y-%m-%d')}")

        fig = go.Figure()
        hv = full_df.tail(90)
        fig.add_trace(go.Scatter(x=hv.index, y=hv['Close'], name='Actual Price', line=dict(color='white', width=2)))
        
        if 'results' in st.session_state:
            st.divider()
            ui.render_performance_summaries(pred_log.get_performance_stats(), clean_df, latest_price_val)
            r = st.session_state['results']
            fig.add_trace(go.Scatter(x=r['backtest'].index, y=r['backtest'].values, name='Model (Backtest)', line=dict(color='cyan', dash='dot', width=1)))
            fig.add_trace(go.Scatter(x=r['dates'], y=r['prices']+r['std'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=r['dates'], y=r['prices']-r['std'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='68% CI'))
            fig.add_trace(go.Scatter(x=r['dates'], y=r['prices'], name='Forecast Mean', line=dict(color='#00ff00', width=3)))
            if st.session_state.get('plan_triggered'):
                fig.add_trace(go.Scatter(x=r['dates'], y=[r['target']]*len(r['dates']), name='Target', line=dict(color='red', dash='dash')))
        
        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='black', plot_bgcolor='black', margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        ui.render_prediction_evaluation_chart(pred_log.get_performance_stats(), full_df, res)
        
        # Signal Impact UI
        if res.get('impact_df') is not None:
            ui.render_signal_attribution_analysis(res['impact_df'])
        elif res.get('impact_df_json'):
            import io
            ui.render_signal_attribution_analysis(pd.read_json(io.StringIO(res['impact_df_json'])))

    with tab2:
        st.subheader("Investment Journal")
        invests = inv_mgr.load_investments()
        if not invests:
            st.info("No active investments tracked in Firestore.")
        else:
            # We use today's forecast to recalculate the status of all saved investments
            current_f_dates = base_res['dates']
            current_f_prices = base_res['prices']
            
            from forecasting_engine import calculate_withdrawal_date

            for inv in reversed(invests):
                roi = ((latest_price_val / inv['price']) - 1) * 100
                target_price = inv['price'] * (1 + inv.get('profit_target', 2.0)/100)
                
                # Calculate Current Projected Exit based on today's forecast
                current_exit = calculate_withdrawal_date(current_f_dates, current_f_prices, target_price)
                original_exit = pd.to_datetime(inv.get('original_withdrawal_date')) if inv.get('original_withdrawal_date') else None
                
                title = f"{inv['date']} | ROI: {roi:.1f}% | Entry: ${inv['price']:,.0f}"
                with st.expander(title):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write(f"**Investment Amount:** ${inv['amount']}")
                        st.write(f"**Target:** {inv.get('profit_target', 2.0)}% (${target_price:,.0f})")
                        
                        if st.button("Delete Record", key=f"del_{inv['id']}", use_container_width=True):
                            inv_mgr.delete_investment(inv['id'])
                            st.rerun()

                    with col2:
                        st.write(f"**Initial Projected Exit:** {original_exit.strftime('%Y-%m-%d') if original_exit else 'N/A'}")
                        if current_exit:
                            st.write(f"**Current Predicted Exit:** {current_exit.strftime('%Y-%m-%d')}")
                            if original_exit:
                                diff = (current_exit - original_exit).days
                                if diff > 0:
                                    st.warning(f"Temporal Shift: +{diff} days (Retreating)")
                                elif diff < 0:
                                    st.success(f"Temporal Shift: {diff} days (Advancing)")
                                else:
                                    st.info("Temporal Shift: Stable")
                        else:
                            st.error("Exit Target not reached in 30-day window.")

                    # --- Dual Line Chart with Shift Markers ---
                    if "forecast_prices" in inv and inv["forecast_prices"]:
                        fd = [pd.to_datetime(d) for d in inv['forecast_dates']]
                        raw_p = np.array(inv.get('forecast_prices'))
                        cal_p = np.array(inv.get('calibrated_prices', inv['forecast_prices']))
                        
                        m_fig = go.Figure()
                        
                        # Actual History since investment
                        asince = full_df[full_df.index >= pd.to_datetime(inv['date'])]
                        if not asince.empty:
                            m_fig.add_trace(go.Scatter(x=asince.index, y=asince['Close'], name='Actual Price', line=dict(color='white', width=2)))
                        
                        # Snapshot: Raw vs Calibrated
                        m_fig.add_trace(go.Scatter(x=fd, y=raw_p, name='Base Model', line=dict(color='rgba(0, 255, 255, 0.4)', dash='dot', width=1)))
                        m_fig.add_trace(go.Scatter(x=fd, y=cal_p, name='Market Aligned', line=dict(color='#00ff00', width=2)))
                        
                        # Target Line
                        m_fig.add_trace(go.Scatter(x=fd, y=[target_price]*len(fd), name='Target', line=dict(color='red', width=1, dash='dash')))
                        
                        # Exit Markers (Vertical Lines)
                        if original_exit:
                            m_fig.add_vline(x=original_exit.timestamp() * 1000, line_width=2, line_dash="dash", line_color="orange", annotation_text="Initial Exit")
                        if current_exit:
                            m_fig.add_vline(x=current_exit.timestamp() * 1000, line_width=2, line_dash="solid", line_color="#00ff00", annotation_text="Current Exit")

                        m_fig.update_layout(height=300, template="plotly_dark", paper_bgcolor='black', plot_bgcolor='black', margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(m_fig, use_container_width=True)
    
    st.divider()
    dv = base_res.get('avg_drift', 0.0)
    if abs(dv) > 10: st.error(f"HIGH DRIFT: {dv:+.1f} pts")
    elif abs(dv) > 5: st.warning(f"MODERATE DRIFT: {dv:+.1f} pts")
    else: st.success(f"STABLE ALIGNMENT: {dv:+.1f} pts")
else:
    st.error("Model assets or data could not be loaded. Please check system logs.")
