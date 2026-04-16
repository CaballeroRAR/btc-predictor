import os
import sys
# Path resolution for industrial architecture
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(root_path, "src")
sys.path.append(root_path)
sys.path.append(src_path)

import streamlit as st
st.set_page_config(layout="wide", page_title="BTC Pulse Predictor")
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# 1. Configuration & Secrets
import src.cloud_config as cloud_config
import src.ui_blocks as ui
from src.utils.logger import setup_logger
from src.facades.forecasting import ForecastingFacade
from src.facades.simulation_facade import SimulationFacade
from src.facades.lifecycle_facade import LifecycleFacade
from src.utils.style_utils import inject_industrial_theme

logger = setup_logger("ui.dashboard")
forecaster = ForecastingFacade()
simulator = SimulationFacade()
lifecycle_manager = LifecycleFacade()
# --- SECURITY GATEWAY ---
def check_password():
    """Returns True if the user had the correct password."""
    if st.session_state.get("authenticated"):
        return True

    PASSWORD = os.getenv("DASHBOARD_PASSWORD")

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
model, scaler = lifecycle_manager.load_model_assets()

# Sidebar: Force Refresh
st.sidebar.header("Global Controls")
force_refresh = st.sidebar.button("Force Market Refresh", width='stretch')

if force_refresh:
    logger.info("User triggered Global Market Refresh. Flushing caches and preserving session context.")
    st.cache_data.clear()
    
    # Store critical session keys to preserve after clear
    auth = st.session_state.get("authenticated")
    nav = st.session_state.get("nav_tab")
    
    st.session_state.clear()
    
    # Restore critical keys and set the pending flag
    if auth: st.session_state["authenticated"] = auth
    if nav: st.session_state["nav_tab"] = nav
    st.session_state["pending_force_refresh"] = True
    
    st.rerun()

# 2. Market Data Orchestration
if 'last_market_sync' not in st.session_state:
    st.session_state['last_market_sync'] = datetime.now() - timedelta(hours=1) # Force first pull

# Check for pending refresh flag
active_force = st.session_state.pop("pending_force_refresh", False) or force_refresh

@st.cache_data
def load_market_data(force=False):
    return lifecycle_manager.load_dataset(force=force)

full_df = load_market_data(force=active_force)
clean_df = full_df # In this architecture, orchestrator handles cleaning

# Inject High-Performance CSS
inject_industrial_theme()

# UI Layout
st.markdown("<h1 style='color: #00ffff; font-family: monospace;'>BTC PULSE PREDICTOR</h1>", unsafe_allow_html=True)
st.caption("Powered by LSTM and Market Psychology | Phase 4 HUD Active")

if model and scaler and not full_df.empty:
    # 3. Sync Actuals (Non-blocking spinner)
    if 'actuals_updated' not in st.session_state:
        with st.spinner("Syncing latest market evaluation..."):
            forecaster.sync_market_actuals(full_df=full_df)
            st.session_state['actuals_updated'] = True

    # 4. Throttled Market Metrics Initialization
    SYNC_WINDOW_MINUTES = 5
    now = datetime.now()
    
    # Check if market data needs refresh
    time_since_market = (now - st.session_state.get('last_market_sync', now - timedelta(hours=1))).total_seconds()
    should_refresh_market = active_force or (time_since_market > SYNC_WINDOW_MINUTES * 60)
    
    if should_refresh_market or 'live_ctx_cache' not in st.session_state:
        with st.spinner("Refreshing Global Market Pulse..."):
            st.session_state['live_ctx_cache'] = forecaster.get_live_market_context()
            st.session_state['last_market_sync'] = now
            # CRITICAL: Invalidate both simulation AND base forecast to prevent desync
            st.session_state.pop('results', None) 
            st.session_state.pop('base_forecast', None)
            st.session_state.pop('plan_triggered', None)

    live_ctx = st.session_state['live_ctx_cache']
    
    # 5. Orchestrate Forecast
    if 'base_forecast' not in st.session_state:
        # If market was just refreshed, we force a fresh neural inference
        force_infer = active_force or should_refresh_market
        with st.spinner("Initializing Latest Market Forecast..."):
            st.session_state['base_forecast'] = forecaster.get_forecast(
                model, scaler, clean_df, 
                force=force_infer, 
                trigger_type="MANUAL_REFRESH" if active_force else "AUTO"
            )
    
    base_res = st.session_state['base_forecast']
    if base_res.get('is_cached'):
        # Fallback to 'timestamp' if calculation_time is missing (for older cache compatibility)
        ts = base_res.get('calculation_time') or base_res.get('timestamp')
        if ts:
            st.caption(f"Loaded from cache (Generated {ts.strftime('%H:%M')} UTC)")
    latest_price_val = live_ctx['live_price']
    interest_pulse = live_ctx['interest_pulse']
    latest_date_val = st.session_state['last_market_sync']
    
    logger.info(f"UI Overlay Updated: Price=${latest_price_val:,.2f} | Pulse={interest_pulse:,.0f}")

    # 6. Sidebar Simulation Controls
    with st.sidebar:
        st.divider()
        st.header("Simulation Settings")
        selected_date = st.date_input("Investment Date", value=latest_date_val.date())
        
        # Calculate auto-price from the selected date
        idx = clean_df.index.get_indexer([pd.to_datetime(selected_date)], method='nearest')[0]
        auto_price = float(clean_df.iloc[idx]['Close'])
        
        # Default to latest price if date is today
        default_entry = latest_price_val if selected_date == latest_date_val.date() else auto_price
        entry_price = st.number_input("Entry Price (USD)", value=default_entry)
        investment = st.number_input("Investment Amount ($)", value=1000)
        profit_target = st.number_input("Profit Target (%)", value=2.0, step=0.5)
        
        # Immediate calculation based on stored base forecast
        if 'results' not in st.session_state:
            st.session_state['results'] = simulator.run_investment_simulation(base_res, auto_price, profit_target, investment)
            st.session_state['plan_triggered'] = False

        if st.button("Run Simulation", width='stretch'):
            st.session_state['results'] = simulator.run_investment_simulation(base_res, entry_price, profit_target, investment)
            st.session_state['plan_triggered'] = True
        
        if st.button("Save this Simulation", width='stretch'):
            r = st.session_state['results']
            simulator.save_to_journal(r)
            st.success("Investment saved!")

        st.divider()
        with st.expander("System Health"):
            status = lifecycle_manager.get_system_status()
            st.write(f"**Model Age:** {status.get('model_age_days', '??')} days")
            if status.get('last_training_date'):
                st.write(f"**Model Updated:** {status['last_training_date'].strftime('%Y-%m-%d')}")
            
            # Display Last Prediction Time
            last_run = base_res.get('calculation_time') or base_res.get('timestamp')
            if last_run:
                st.write(f"**Last Prediction:** {last_run.strftime('%H:%M:%S UTC')}")
            
            # Use the drift directly from our results
            current_drift = base_res.get('avg_drift', 0.0)
            st.write(f"**Drift:** {current_drift:+.2f} pts")
            
            if st.button("Recalibrate Market Alignment", width='stretch'):
                with st.spinner("Calibrating Neural Engine (300 Step Optimization)..."):
                    now = datetime.now()
                    time_since_sync = (now - st.session_state['last_market_sync']).total_seconds() / 60
                    
                    if time_since_sync > 30:
                        # Stale Data Path: Full historical rescan
                        logger.info(f"Smart Recalibrate: Data is stale ({time_since_sync:.1f}m). Triggering full market scan.")
                        _, f_clean = load_market_data(force=True)
                        st.session_state['last_market_sync'] = now
                    else:
                        # Fast Path: Data is fresh, only pull latest price via facade internal
                        logger.info(f"Smart Recalibrate: Data is fresh ({time_since_sync:.1f}m). Using fast-track recalibration.")
                        f_clean = clean_df
                    
                    st.session_state['base_forecast'] = forecaster.get_forecast(
                        model, scaler, f_clean, 
                        force=True, 
                        trigger_type="MANUAL_REFRESH"
                    )
                    st.rerun()

        st.divider()
        with st.expander("Model Management"):
            # 1. Job Telemetry (Throttled Monitoring)
            st.markdown("### Training Status")
            
            # Sub-telemetry gate: Auto-refresh every 5 mins OR manual override
            time_since_job = (now - st.session_state.get('last_job_sync_time', now - timedelta(hours=1))).total_seconds()
            should_refresh_job = st.session_state.get('force_job_refresh') or (time_since_job > 300)
            
            if 'active_job_cache' not in st.session_state or should_refresh_job:
                with st.spinner("Fetching Vertex AI Jobs..."):
                    st.session_state['active_job_cache'] = lifecycle_manager.get_active_training_jobs(limit=1)
                    st.session_state['force_job_refresh'] = False
                    st.session_state['last_job_sync_time'] = now
            
            active_job = st.session_state['active_job_cache']
            
            if active_job:
                st.info(active_job["status"])
                if st.button("Refresh Job Status", width='stretch'):
                    st.session_state['force_job_refresh'] = True
                    st.rerun()
            else:
                st.write("No active training jobs found in scanned regions.")
                if st.button("Check for Jobs", width='stretch'):
                    st.session_state['force_job_refresh'] = True
                    st.rerun()

            st.markdown("### Asset Management")
            if st.button("Synchronize Model Assets", width='stretch'):
                with st.spinner("Downloading assets from GCS..."):
                    if lifecycle_manager.sync_assets(force=True):
                        st.success("Model assets synchronized. Analytical core updated.")
                        st.rerun()
                    else:
                        st.error("GCS Synchronization failed.")

            st.divider()
            # 2. Training Trigger
            st.markdown("### Retraining")
            st.warning("Launching a new training job will provision a GCP compute instance. This will incur cloud costs.")
            
            confirm_training = st.checkbox("Confirm Cloud Training Trigger")
            if st.button("Launch Vertex AI Model Training", disabled=not confirm_training, width='stretch'):
                with st.spinner("Provisioning Vertex AI CustomJob..."):
                    try:
                        lifecycle_manager.launch_retraining()
                        st.success("Vertex AI Training Job successfully triggered!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to launch job: {str(e)}")

    # Tab Navigation (Pinned to session state to prevent refresh reset)
    tabs = ["Market Analysis", "Investment Journal"]
    if 'nav_tab' not in st.session_state:
        st.session_state['nav_tab'] = tabs[0]
    
    active_tab = st.radio(
        "Navigation", 
        tabs, 
        key='nav_tab',
        horizontal=True, 
        label_visibility="collapsed"
    )
    
    if active_tab == "Market Analysis":
        res = st.session_state['base_forecast']
        
        # Throttled retrieval of the latest manual pulse baseline
        if 'manual_snapshot' not in st.session_state or should_refresh_market:
            with st.spinner("Retrieving Manual Baseline..."):
                st.session_state['manual_snapshot'] = forecaster.firestore_repo.get_latest_manual_snapshot()
        
        man_res = st.session_state.get('manual_snapshot')
        
        # Display summary with the live price and curiosity pulse
        ui.render_market_summary_metrics(
            latest_price_val, 
            latest_date_val, 
            float(res['prices'][0]), 
            res['dates'][0].strftime('%Y-%m-%d'),
            interest_pulse=interest_pulse
        )
        
        if st.session_state.get('plan_triggered'):
            res_obj = st.session_state['results']
            ui.render_confidence_indicator(res_obj.confidence_score, res_obj.simulation_status)
            if res_obj.projected_withdrawal_date:
                st.success(f"Predicted Profit Target Hit Date: {res_obj.projected_withdrawal_date}")

        fig = go.Figure()
        # Visual focus: last 90 days + live tick (Normalized to today for grid alignment)
        hv = full_df.tail(90).copy()
        # Ensure we anchor the live price to the midnight-normalized 'today' to sit on the gridline
        live_t_normalized = pd.to_datetime(latest_date_val).normalize()
        live_entry = pd.DataFrame({'Close': [latest_price_val]}, index=[live_t_normalized])
        hv = pd.concat([hv, live_entry])
        
        fig.add_trace(go.Scatter(
            x=hv.index, y=hv['Close'], 
            name='Actual Price', 
            mode='lines+markers',
            line=dict(color='white', width=2),
            marker=dict(size=6)
        ))
        
        if 'results' in st.session_state:
            st.divider()
            r = st.session_state['results']
            
            # Note: Backtest is in 'res' (base_forecast), not in 'r' (simulation)
            fig.add_trace(go.Scatter(x=res['backtest'].index, y=res['backtest'].values, name='Model (Backtest)', line=dict(color='cyan', dash='dot', width=1)))
            
            # 1. Latest Manual Refresh Baseline (Static Baseline)
            if man_res:
                man_dt = man_res.get('timestamp')
                man_str = man_dt.strftime('%m/%d %H:%M') if isinstance(man_dt, datetime) else "N/A"
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(d) for d in man_res['dates']], 
                    y=man_res['prices'],
                    name=f'Latest Refresh State ({man_str})',
                    mode='lines',
                    line=dict(color='#ffaa00', width=4.5, dash='dot'), # Thicker dotted line to peek through
                    opacity=0.8
                ))

            # 2. Confidence Interval (Today's Active Model)
            fig.add_trace(go.Scatter(x=r.forecast_dates, y=np.array(r.forecast_prices)+np.array(r.std), mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=r.forecast_dates, y=np.array(r.forecast_prices)-np.array(r.std), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='68% CI'))
            
            # 3. Forecast Mean (Today's Active Model)
            res_dt = res.get('calculation_time') or res.get('timestamp')
            res_str = res_dt.strftime('%H:%M') if isinstance(res_dt, datetime) else "Live"
            
            fig.add_trace(go.Scatter(
                x=r.forecast_dates, y=r.forecast_prices, 
                name=f'Forecast Mean ({res_str})', 
                mode='lines+markers',
                line=dict(color='#00ff00', width=3),
                marker=dict(size=8, symbol='circle')
            ))
            if st.session_state.get('plan_triggered'):
                target_val = r.target_price
                fig.add_trace(go.Scatter(x=r.forecast_dates, y=[target_val]*len(r.forecast_dates), name='Target', line=dict(color='red', dash='dash')))

            # --- ENTRY SNAPSHOT OVERLAY ---
            investments = simulator.get_journal_entries()
            if investments:
                # Sort by created_at or timestamp to find the most recent snapshot
                latest_inv = sorted(investments, key=lambda x: x.get('created_at', ''), reverse=True)[0]
                
                if 'forecast_prices' in latest_inv and 'forecast_dates' in latest_inv:
                    e_dates = [pd.to_datetime(d) for d in latest_inv['forecast_dates']]
                    e_prices = latest_inv['forecast_prices']
                    
                    # Format display date
                    snap_dt = latest_inv.get('created_at')
                    snap_str = snap_dt.strftime('%m/%d %H:%M') if isinstance(snap_dt, datetime) else "Unknown"
                    
                    fig.add_trace(go.Scatter(
                        x=e_dates, y=e_prices, 
                        name=f'Last Entry State ({snap_str})', 
                        mode='lines',
                        line=dict(color='rgba(0, 200, 255, 0.6)', width=1.5, dash='dot')
                    ))
        
        fig.update_layout(
            height=500, 
            template="plotly_dark", 
            paper_bgcolor='black', 
            plot_bgcolor='black', 
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Consistent Grid Hygiene & Viewport Control (Midnight Normalized)
        now = datetime.now()
        today_dt = datetime(now.year, now.month, now.day)
        view_start = today_dt - timedelta(days=30)
        
        # Determine view_end and ensure normalization
        raw_end = r.forecast_dates[-1] if 'results' in st.session_state else today_dt + timedelta(days=30)
        view_end = pd.to_datetime(raw_end).normalize()
        
        # Incremental Day View (Daily - Snap to Midnight)
        all_dates = pd.date_range(start=view_start, end=view_end, freq='D')
        tick_text = [f"{d.day}{d.strftime('%a')[0].upper()}" for d in all_dates]
        
        grid_style = dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
        fig.update_xaxes(
            range=[view_start, view_end],
            tickmode='array',
            tickvals=all_dates,
            ticktext=tick_text,
            **grid_style
        )
        # Highlight Sundays for weekly grouping
        for d in all_dates:
            if d.weekday() == 6: # Sunday
                fig.add_vline(x=d, line_width=1, line_dash="dot", line_color="rgba(255, 255, 255, 0.35)")
                
        fig.update_yaxes(title_text="Price (USD)", **grid_style)
        
        st.plotly_chart(fig, width='stretch')
        ui.render_prediction_evaluation_chart(forecaster.get_performance_history(), full_df, latest_price_val, res)
        
        # Signal Impact UI
        if res.get('impact_df') is not None:
            ui.render_signal_attribution_analysis(res['impact_df'])
        elif res.get('impact_df_json'):
            import io
            ui.render_signal_attribution_analysis(pd.read_json(io.StringIO(res['impact_df_json'])))

    else: # Investment Journal
        st.subheader("Investment Journal")
        
        # Throttled Journal Retrieval
        time_since_journal = (now - st.session_state.get('last_journal_sync', now - timedelta(hours=1))).total_seconds()
        should_refresh_journal = force_refresh or (time_since_journal > SYNC_WINDOW_MINUTES * 60)
        
        if should_refresh_journal or 'investments_cache' not in st.session_state:
            with st.spinner("Syncing Investment Journal..."):
                st.session_state['investments_cache'] = simulator.get_journal_entries()
                st.session_state['last_journal_sync'] = now

        invests = st.session_state['investments_cache']
        
        if not invests:
            st.info("No active investments tracked in Firestore.")
        else:
            # We use today's forecast to recalculate the status of all saved investments
            current_f_dates = base_res['dates']
            current_f_prices = base_res['prices']
            
            from src.core.simulation import calculate_withdrawal_date

            # latest_price_val is already defined at the start of the UI loop from full_df
            for inv in reversed(invests):
                # Format date for display
                display_date = inv.get('created_at')
                if isinstance(display_date, datetime):
                    display_date = display_date.strftime('%Y-%m-%d')
                
                roi = ((latest_price_val / inv['entry_price']) - 1) * 100
                target_price = inv['target_price']
                
                # Calculate Current Projected Exit based on today's forecast
                current_exit = calculate_withdrawal_date(current_f_dates, current_f_prices, target_price)
                original_exit = pd.to_datetime(inv.get('projected_withdrawal_date')) if inv.get('projected_withdrawal_date') else None
                
                title = f"INVESTMENT: {display_date} (ROI: {roi:.1f}%)"
                with st.expander(title):
                    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
                    # --- PREMIUM METADATA GRID ---
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: 
                        st.caption("Capital Invested")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>${inv['amount']}</h3>", unsafe_allow_html=True)
                    with m2: 
                        st.caption("Target ROI (%)")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>{inv.get('target_pct', 2.0)}%</h3>", unsafe_allow_html=True)
                    with m3: 
                        st.caption("Entry Price")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>${inv['entry_price']:,.0f}</h3>", unsafe_allow_html=True)
                    with m4: 
                        st.caption("Target Exit Price")
                        st.markdown(f"<h3 style='color: #00ffff; margin-top: -20px;'>${inv['target_price']:,.0f}</h3>", unsafe_allow_html=True)
                    
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
                        asince = full_df[full_df.index >= pd.to_datetime(display_date)].tail(15)
                        if not asince.empty:
                            m_fig.add_trace(go.Scatter(
                                x=asince.index, y=asince['Close'], 
                                name='Actual Price', 
                                line=dict(color='rgba(255, 255, 255, 0.6)', width=1.5)
                            ))
                        
                        # Snapshot: Raw vs Calibrated (Now "Market Prediction")
                        today_dt = datetime.now().date()
                        
                        # Past Prediction: Lower Alpha (0.4)
                        past_mask = [d.date() < today_dt for d in fd]
                        p_fd = [d for d, m in zip(fd, past_mask) if m]
                        p_cal_p = [p for p, m in zip(cal_p, past_mask) if m]
                        
                        if p_fd:
                            m_fig.add_trace(go.Scatter(
                                x=p_fd, y=p_cal_p, 
                                name='Market Prediction (Past)', 
                                mode='lines+markers', 
                                line=dict(color='rgba(0, 255, 255, 0.4)', width=1.5), 
                                marker=dict(size=7, symbol='diamond')
                            ))

                        # Future Prediction: Solid (1.0)
                        future_mask = [d.date() >= today_dt for d in fd]
                        f_fd = [d for d, m in zip(fd, future_mask) if m]
                        f_cal_p = [p for p, m in zip(cal_p, future_mask) if m]
                        
                        if f_fd:
                            m_fig.add_trace(go.Scatter(
                                x=f_fd, y=f_cal_p, 
                                name='Market Prediction (Future)', 
                                mode='lines+markers', 
                                line=dict(color='#00ffff', width=2.5), 
                                marker=dict(size=7, symbol='diamond')
                            ))
                        
                        # --- LIVE OVERLAY (The Fix) ---
                        # Overlay the current global forecast dates/prices onto this investment's chart
                        m_fig.add_trace(go.Scatter(
                            x=current_f_dates, 
                            y=current_f_prices, 
                            name='Live Global Forecast', 
                            mode='lines', 
                            line=dict(color='#00ffff', width=1, dash='dot')
                        ))
                        
                        # --- TEMPORAL DRIFT BRIDGE ---
                        # Draw vertical lines between original diamonds and live forecast for dates before yesterday
                        yesterday_dt = today_dt - timedelta(days=1)
                        bridge_x = []
                        bridge_y = []
                        live_lookup = {d.date(): p for d, p in zip(current_f_dates, current_f_prices)}
                        
                        for d, p in zip(p_fd, p_cal_p):
                            if d.date() < yesterday_dt and d.date() in live_lookup:
                                bridge_x.extend([d, d, None]) # [Start, End, Break]
                                bridge_y.extend([p, live_lookup[d.date()], None])
                        
                        if bridge_x:
                            m_fig.add_trace(go.Scatter(
                                x=bridge_x, y=bridge_y,
                                name='Drift Bridge',
                                mode='lines',
                                line=dict(color='rgba(0, 255, 255, 0.2)', width=1, dash='dot'),
                                showlegend=False,
                                hoverinfo='skip'
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
                                showarrow=False, font=dict(color='yellow', size=13), 
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
                                showarrow=False, font=dict(color='#00ffff', size=13), 
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
                        st.plotly_chart(m_fig, width="stretch")
                        
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
                            confirm_key = f"confirm_del_{inv['id']}"
                            if st.session_state.get(confirm_key):
                                st.markdown("<p style='color: #ff4b4b; font-size: 0.8em; text-align: right; margin-bottom: 5px;'>Destructive Action: Are you sure?</p>", unsafe_allow_html=True)
                                sub_col1, sub_col2 = st.columns(2)
                                with sub_col1:
                                    if st.button("Cancel", key=f"can_{inv['id']}"):
                                        st.session_state[confirm_key] = False
                                        st.rerun()
                                with sub_col2:
                                    if st.button("Purge Entry", key=f"del_{inv['id']}", use_container_width=True):
                                        simulator.delete_entry(inv['id'])
                                        st.rerun()
                            else:
                                if st.button("Delete Record", key=f"del_{inv['id']}", type="secondary"):
                                    st.session_state[confirm_key] = True
                                    st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # Infrastructure Audit
    st.divider()
    with st.expander("Analytical Infrastructure Inventory"):
        sys_col1, sys_col2, sys_col3 = st.columns(3)
        sys_col1.write(f"**GCP Project:** `{cloud_config.PROJECT_ID}`")
        sys_col2.write(f"**Computation Region:** `{cloud_config.REGION}`")
        sys_col3.write(f"**Model Context:** `LOCAL_LSTM_V2`")
        st.caption("Verify environment alignment by cross-referencing this Project ID with your Google Cloud Console URL.")

    st.divider()
    dv = base_res.get('avg_drift', 0.0)
    if abs(dv) > 10: st.error(f"HIGH DRIFT: {dv:+.1f} pts")
    elif abs(dv) > 5: st.warning(f"MODERATE DRIFT: {dv:+.1f} pts")
    else: st.success(f"STABLE ALIGNMENT: {dv:+.1f} pts")
else:
    st.error("Model assets or data could not be loaded. Please check system logs.")
