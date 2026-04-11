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
            if st.form_submit_button("Login", width='stretch'):
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
model, scaler = lifecycle.get_active_model()

# Sidebar: Force Refresh
st.sidebar.header("Global Controls")
force_refresh = st.sidebar.button("Force Market Refresh", width='stretch')

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

        if st.button("Run Simulation", width='stretch'):
            st.session_state['results'] = logic.calculate_withdrawal_plan(base_res, entry_price, profit_target, investment)
            st.session_state['plan_triggered'] = True
        
        if st.button("Save this Simulation", width='stretch'):
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
            
            if st.button("Recalibrate Market Alignment", width='stretch'):
                # Force a fresh market data pull to ensure latest ticker prices are used for drift analysis
                _, f_clean = load_market_data(force=True)
                st.session_state['base_forecast'] = logic.get_base_forecast(db_mgr, model, scaler, f_clean, force=True)
                st.session_state['active_tab'] = "Investment Journal"
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
                if st.button("Refresh Job Status", width='stretch'):
                    st.rerun()
            else:
                st.write("No active training jobs found in scanned regions.")

            st.markdown("### Asset Management")
            if st.button("🔄 Sync/Update Model from Cloud Storage", width='stretch', help="Pull the latest .h5 and .pkl files from GCS. Use this once your Vertex Job is SUCCEEDED."):
                with st.spinner("Downloading assets from GCS..."):
                    lifecycle.force_sync_from_gcs(check_exists=False)
                    st.success("Model assets updated! Refreshing analytical core...")
                    st.rerun()

            st.divider()
            # 2. Training Trigger
            st.markdown("### Retraining")
            st.warning("Launching a new training job will provision a GCP compute instance. This will incur cloud costs.")
            
            confirm_training = st.checkbox("Confirm Cloud Training Trigger")
            if st.button("Launch Vertex AI Model Training", disabled=not confirm_training, width='stretch'):
                with st.spinner("Provisioning Vertex AI CustomJob..."):
                    try:
                        new_job = vertex.trigger_training_job()
                        st.success("Vertex AI Training Job successfully triggered!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to launch job: {str(e)}")

    # 6. Main Visualizations (Persistent Navigation)
    tabs = ["Market Analysis", "Investment Journal"]
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = tabs[0]
        
    latest_price_val = full_df['Close'].iloc[-1]
    latest_date_val = full_df.index[-1]
    
    active_tab = st.radio("Navigation", tabs, index=tabs.index(st.session_state['active_tab']), horizontal=True, label_visibility="collapsed")
    st.session_state['active_tab'] = active_tab
    
    if active_tab == "Market Analysis":
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
        st.plotly_chart(fig, width='stretch')
        ui.render_prediction_evaluation_chart(pred_log.get_performance_stats(), full_df, res)
        
        # Signal Impact UI
        if res.get('impact_df') is not None:
            ui.render_signal_attribution_analysis(res['impact_df'])
        elif res.get('impact_df_json'):
            import io
            ui.render_signal_attribution_analysis(pd.read_json(io.StringIO(res['impact_df_json'])))

    else: # Investment Journal
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
                    # --- PREMIUM METADATA GRID ---
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: 
                        st.caption("Capital Invested")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>${inv['amount']}</h3>", unsafe_allow_html=True)
                    with m2: 
                        st.caption("Target ROI (%)")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>{inv.get('profit_target', 2.0)}%</h3>", unsafe_allow_html=True)
                    with m3: 
                        st.caption("Entry Price")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>${inv['price']:,.0f}</h3>", unsafe_allow_html=True)
                    with m4: 
                        st.caption("Target Exit Price")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>${target_price:,.0f}</h3>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # --- TEMPORAL AUDIT ---
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.caption("Initial Exit Projection")
                        st.write(f"**{original_exit.strftime('%Y-%m-%d') if original_exit else 'N/A'}**")
                    with sub2:
                        st.caption("Current Live Projection")
                        if current_exit:
                            st.write(f"**{current_exit.strftime('%Y-%m-%d')}**")
                        else:
                            st.write("Out of Window")

                    # --- Dual Line Chart with Shift Markers ---
                    if "forecast_prices" in inv and inv["forecast_prices"]:
                        fd = [pd.to_datetime(d) for d in inv['forecast_dates']]
                        raw_p = np.array(inv.get('forecast_prices'))
                        cal_p = np.array(inv.get('calibrated_prices', inv['forecast_prices']))
                        
                        m_fig = go.Figure()
                        
                        # Actual History since investment (Trimmed to last 15 days)
                        asince = full_df[full_df.index >= pd.to_datetime(inv['date'])].tail(15)
                        if not asince.empty:
                            m_fig.add_trace(go.Scatter(x=asince.index, y=asince['Close'], name='Actual Price', line=dict(color='white', width=2)))
                        
                        # Snapshot: Raw vs Calibrated (Now "Market Prediction")
                        m_fig.add_trace(go.Scatter(x=fd, y=cal_p, name='Market Prediction', mode='lines+markers', line=dict(color='#00ffff', width=2), marker=dict(size=6)))
                        
                        # --- LIVE OVERLAY (The Fix) ---
                        # Overlay the current global forecast dates/prices onto this investment's chart
                        m_fig.add_trace(go.Scatter(
                            x=current_f_dates, 
                            y=current_f_prices, 
                            name='Live Global Forecast', 
                            mode='lines', 
                            line=dict(color='#00ffff', width=1, dash='dot')
                        ))
                        
                        # Target Line
                        m_fig.add_trace(go.Scatter(x=fd, y=[target_price]*len(fd), name='Target', line=dict(color='red', width=1, dash='dash')))
                        
                        # Exit Markers (Vertical Lines)
                        if original_exit:
                            x_val = original_exit.timestamp() * 1000
                            m_fig.add_shape(
                                type='line', x0=x_val, x1=x_val, y0=0, y1=0.5, yref='paper',
                                line=dict(color='yellow', dash='dot', width=2)
                            )
                            m_fig.add_annotation(
                                x=x_val, y=0.1, yref='paper', text="Initial Exit", 
                                showarrow=False, font=dict(color='yellow', size=10), 
                                textangle=-90, xanchor='left', xshift=10
                            )
                        if current_exit:
                            x_val = current_exit.timestamp() * 1000
                            m_fig.add_shape(
                                type='line', x0=x_val, x1=x_val, y0=0.5, y1=1, yref='paper',
                                line=dict(color='#00ff00', dash='solid', width=2)
                            )
                            m_fig.add_annotation(
                                x=x_val, y=0.9, yref='paper', text="Current Exit", 
                                showarrow=False, font=dict(color='#00ffff', size=10), 
                                textangle=-90, xanchor='left', xshift=10
                            )

                        # Custom High-Res X-Axis with (+N) Days
                        tick_vals = fd
                        tick_text = [d.strftime('%a %d') for d in fd]
                        
                        m_fig.update_layout(
                            height=300, 
                            template="plotly_dark", 
                            paper_bgcolor='black', 
                            plot_bgcolor='black', 
                            margin=dict(l=0, r=0, t=20, b=0),
                            xaxis=dict(
                                tickmode='array',
                                tickvals=tick_vals,
                                ticktext=tick_text,
                                showgrid=True,
                                gridcolor='rgba(255, 255, 255, 0.1)'
                            )
                        )
                        st.plotly_chart(m_fig, width='stretch')
                        
                        # --- FOOTER: STATUS & RED ACTION ---
                        footer_col1, footer_col2 = st.columns([2, 1])
                        with footer_col1:
                            if current_exit and original_exit:
                                diff = (current_exit - original_exit).days
                                status_msg = "Stable" if diff == 0 else (f"+{diff} days Retreating" if diff > 0 else f"{diff} days Advancing")
                                st.caption(f"Temporal Shift: {status_msg}")
                            elif not current_exit:
                                st.caption("Temporal Shift: Out of Window")
                        
                        with footer_col2:
                            st.markdown("<p style='color: #ff4b4b; font-size: 0.8em; text-align: right; margin-bottom: -5px;'>Destructive Action</p>", unsafe_allow_html=True)
                            if st.button("Delete Record", key=f"del_{inv['id']}", width='stretch', type="secondary"):
                                inv_mgr.delete_investment(inv['id'])
                                st.rerun()
    
    # Infrastructure HUD
    st.divider()
    with st.expander("System Infrastructure Status (Admin Only)"):
        sys_col1, sys_col2, sys_col3 = st.columns(3)
        sys_col1.write(f"**GCP Project:** `{cloud_config.PROJECT_ID}`")
        sys_col2.write(f"**Target Region:** `{cloud_config.REGION}`")
        sys_col3.write(f"**Model Context:** `LOCAL_LSTM_V1`")
        st.caption("Verify this Project ID matches your Google Cloud URL to ensure you are viewing the correct console.")

    st.divider()
    dv = base_res.get('avg_drift', 0.0)
    if abs(dv) > 10: st.error(f"HIGH DRIFT: {dv:+.1f} pts")
    elif abs(dv) > 5: st.warning(f"MODERATE DRIFT: {dv:+.1f} pts")
    else: st.success(f"STABLE ALIGNMENT: {dv:+.1f} pts")
else:
    st.error("Model assets or data could not be loaded. Please check system logs.")
