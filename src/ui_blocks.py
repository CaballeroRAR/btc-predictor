import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def render_market_summary_metrics(latest_price_val, latest_date_val, forecast_today_val, forecast_today_date):
    """Render the Live Price vs Forecast metrics."""
    st.subheader("Market Summary")
    mcols = st.columns(3)
    mcols[0].metric(f"Live BTC ({latest_date_val.strftime('%H:%M')})", f"${latest_price_val:,.2f}")
    
    if forecast_today_val:
        diff = latest_price_val - forecast_today_val
        diff_pct = (diff / forecast_today_val) * 100
        mcols[1].metric(f"Forecast ({forecast_today_date})", f"${forecast_today_val:,.2f}")
        mcols[2].metric("Difference", f"${diff:,.2f}", f"{diff_pct:+.2f}% vs Forecast", delta_color="off")
    else:
        mcols[1].write("**Forecast:** Loading...")
        mcols[2].write("**Difference:** Pending")

def render_signal_attribution_analysis(impact_df):
    """Render the Signal Attribution charting and evaluation."""
    if impact_df is not None and not impact_df.empty:
        st.divider()
        st.subheader("Signal Attribution & Impact Analysis")
        
        # Metrics for quick scan
        cols = st.columns(3)
        for i, (name, row) in enumerate(impact_df.iterrows()):
            col_idx = i % 3
            cols[col_idx].metric(
                row['Signal Group'], 
                f"${row['USD Impact']:,.0f}", 
                delta=f"{row['Relative Importance']:.2f}% Weight",
                delta_color="off"
            )
        
        col_chart, col_text = st.columns([2, 1])
        
        with col_chart:
            # Sensitivity Bar Chart
            fig_impact = px.bar(
                impact_df,
                x="USD Impact",
                y="Signal Group",
                orientation='h',
                color="Direction",
                color_discrete_map={
                    "Bullish Influence": "#00ff00",
                    "Bearish Influence": "#ff4b4b"
                },
                title="Price Sensitivity per Factor Group",
                text_auto='.2s'
            )
            fig_impact.update_layout(
                template="plotly_dark",
                paper_bgcolor='black',
                plot_bgcolor='black',
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_impact, use_container_width=True)
        
        with col_text:
            st.write("**Evaluation Summary:**")
            top_signal = impact_df.loc[impact_df['Impact Magnitude'].idxmax()]
            st.info(
                f"The **{top_signal['Signal Group']}** group currently has the highest appropriate weight "
                f"in the model, contributing a ${top_signal['Impact Magnitude']:,.2f} deviation "
                f"to the forecast."
            )
            
            if abs(top_signal['USD Impact']) > 1000:
                st.caption("Significant alpha detected in this signal group.")
            else:
                st.caption("Model is currently maintaining neutral signal sensitivity.")

def render_performance_summaries(history_df, clean_df, latest_price_val):
    """Render the historical accuracy and today's session snapshot."""
    if history_df.empty:
        st.info("Run a simulation today to populate live metrics.")
        return

    # 1. Historical Performance (Yesterday D-1)
    yesterday_date = clean_df.index[-1].strftime('%Y-%m-%d')
    actual_yesterday = clean_df['Close'].iloc[-1]
    
    day_before_yesterday = (clean_df.index[-1] - timedelta(days=1)).strftime('%Y-%m-%d')
    pred_yesterday = history_df[
        (history_df['sim_run_date'] == day_before_yesterday) & 
        (history_df['forecast_date'] == yesterday_date)
    ]['predicted_price']
    
    # 2. Live Today Snapshot (D)
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
            st.info("No simulations recorded for today yet.")
