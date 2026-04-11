import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def render_market_summary_metrics(latest_price_val, latest_date_val, forecast_today_val, forecast_today_date):
    """Render the Live Price vs Forecast metrics in a wide grid."""
    st.subheader("Market Summary")
    mcols = st.columns(4)
    mcols[0].metric(f"Live BTC ({latest_date_val.strftime('%H:%M')})", f"${latest_price_val:,.2f}")
    
    if forecast_today_val:
        diff = latest_price_val - forecast_today_val
        diff_pct = (diff / forecast_today_val) * 100
        mcols[1].metric(f"Forecast ({forecast_today_date})", f"${forecast_today_val:,.2f}")
        mcols[2].metric("USD Deviation", f"${diff:,.2f}", delta_color="off")
        mcols[3].metric("Percent Error", f"{diff_pct:+.2f}%", delta_color="inverse")
    else:
        mcols[1].write("**Forecast:** Pending Refresh")

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

def render_prediction_evaluation_chart(history_df, full_df, live_res=None):
    """
    Render a split-logic accuracy chart:
    1. Historical Audit (Today-1 and older): Static lines from DB + full_df closes.
    2. Tactical Monitor (Today): Live Pulsating Dot for active simulation results.
    """
    if full_df.empty:
        return

    st.divider()
    st.subheader("Model Accuracy: Predicted vs. Actual Closing Prices")
    
    today_date = datetime.now().date()
    
    # --- 1. HISTORICAL DATA PREPARATION (Date < Today) ---
    # We take actual closes from the live Market Feed (full_df)
    hist_actuals = full_df[full_df.index.date < today_date].copy()
    hist_actuals = hist_actuals.tail(7) 
    
    # We take matched predictions from the Database (history)
    # Group by date to get the mean prediction for each past day
    if not history_df.empty:
        history_df['forecast_date'] = pd.to_datetime(history_df['forecast_date'])
        hist_preds = history_df[history_df['forecast_date'].dt.date < today_date].copy()
        hist_preds_grouped = hist_preds.groupby(hist_preds['forecast_date'].dt.date)['predicted_price'].mean()
    else:
        hist_preds_grouped = pd.Series(dtype=float)

    # Create the unified historical evaluation dataframe
    eval_df = pd.DataFrame(index=hist_actuals.index.date)
    eval_df['actual_price'] = hist_actuals['Close'].values
    # Map predictions to the dates we have actuals for
    eval_df['predicted_price'] = eval_df.index.map(hist_preds_grouped)
    
    if eval_df.empty:
        st.info("Market feed history is initializing...")
    # --- 2. LIVE DATA PREPARATION (Today) ---
    live_price = full_df['Close'].iloc[-1]
    # In live_res, dates[0] is Today's prediction
    live_pred = live_res['prices'][0] if (live_res and 'prices' in live_res) else None
    
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # LAYER 1: HISTORICAL AUDIT (Static Lines)
    if not eval_df.empty:
        # Calculate error only where we have both prices
        eval_df['error_pct'] = ((eval_df['predicted_price'] / eval_df['actual_price']) - 1) * 100
        
        # Always plot the Actual Market Line
        fig.add_trace(go.Scatter(
            x=eval_df.index, y=eval_df['actual_price'],
            mode='lines+markers', name='Historical Actual',
            line=dict(color='#00ff00', width=2)
        ), secondary_y=False)

        # Plot Predicted Mean only where it exists
        clean_preds = eval_df.dropna(subset=['predicted_price'])
        if not clean_preds.empty:
            fig.add_trace(go.Scatter(
                x=clean_preds.index, y=clean_preds['predicted_price'],
                mode='lines+markers', name='Historical Mean Pred',
                line=dict(color='#ff9900', width=1, dash='dot')
            ), secondary_y=False)

            fig.add_trace(go.Bar(
                x=clean_preds.index, y=clean_preds['error_pct'],
                name='Historical Error %',
                marker_color='rgba(255, 255, 255, 0.1)',
                hovertemplate='%{y:.2f}% Error<extra></extra>'
            ), secondary_y=True)

    # LAYER 2: TACTICAL MONITOR (Live Pulsar for Today)
    # 1. Today's Market Price (Static Dot)
    fig.add_trace(go.Scatter(
        x=[today_date], y=[live_price],
        mode='markers', name='Live Price',
        marker=dict(color='#00ff00', size=10, symbol='circle')
    ), secondary_y=False)

    # 2. Today's Prediction (Pulsating Dot) & Live Deviation
    if live_pred:
        # Calculate Live Deviation %
        live_error_pct = ((live_pred / live_price) - 1) * 100
        
        # Add bar for today's live deviation
        fig.add_trace(go.Bar(
            x=[today_date], y=[live_error_pct],
            name='Live Deviation %',
            marker_color='rgba(0, 255, 255, 0.3)',
            hovertemplate='Live: %{y:.2f}% Error<extra></extra>'
        ), secondary_y=True)

        # Outer Halo (Pulsar effect)
        fig.add_trace(go.Scatter(
            x=[today_date], y=[live_pred],
            mode='markers', name='Live Target (Pulsar)',
            marker=dict(color='rgba(0, 255, 255, 0.2)', size=25, symbol='circle')
        ), secondary_y=False)
        # Inner Core
        fig.add_trace(go.Scatter(
            x=[today_date], y=[live_pred],
            mode='markers', showlegend=False,
            marker=dict(color='#00ffff', size=10, symbol='diamond')
        ), secondary_y=False)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor='black', plot_bgcolor='black',
        height=450, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Timeline",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Grid and Axis Hygiene
    grid_style = dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.15)')
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False, **grid_style)
    fig.update_yaxes(
        title_text="Deviation (%)", 
        secondary_y=True, 
        showgrid=False, 
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.5)', 
        zerolinewidth=1,
        # Symmetrical centering for stable origin reference
        range=[-15, 15] 
    )

    st.plotly_chart(fig, use_container_width=True)

    # Percentage Difference Stats Row
    audit_df = eval_df.dropna(subset=['predicted_price'])
    if not audit_df.empty:
        st.write("**Daily Deviation Audit**")
        cols = st.columns(len(audit_df))
        for i, (date_idx, row) in enumerate(audit_df.iterrows()):
            cols[i].metric(
                date_idx.strftime('%m-%d'), 
                f"{row['error_pct']:+.1f}%",
                help=f"Actual: ${row['actual_price']:,.0f}\nPredicted: ${row['predicted_price']:,.0f}"
            )

    # --- Distribution Drill-Down Section ---
    st.divider()
    st.subheader("Statistical Prediction Audit")
    
    # Get dates that have both predictions and actuals for audit
    audit_dates = sorted(eval_df['forecast_date'].unique(), reverse=True)
    
    if not audit_dates:
        st.info("No audited dates available for statistical drill-down yet.")
        return

    selected_audit_date = st.selectbox(
        "Select Historical Date to Audit Prediction Multiplicity",
        options=audit_dates,
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )

    if selected_audit_date:
        # Filter all historical predictions for that specific day
        # (This includes results from all simulations run before that date)
        dist_df = history_df[pd.to_datetime(history_df['forecast_date']) == selected_audit_date].copy()
        
        if not dist_df.empty:
            actual_for_day = dist_df['actual_price'].dropna().unique()
            actual_val = actual_for_day[0] if len(actual_for_day) > 0 else None
            
            fig_dist = px.histogram(
                dist_df, 
                x="predicted_price", 
                nbins=20,
                title=f"Predicted Price Distribution for {selected_audit_date.strftime('%Y-%m-%d')}",
                color_discrete_sequence=['#ff9900']
            )
            
            if actual_val:
                fig_dist.add_vline(
                    x=actual_val, 
                    line_width=3, 
                    line_dash="dash", 
                    line_color="#00ff00",
                    annotation_text=f"Actual Close: ${actual_val:,.0f}"
                )
            
            fig_dist.update_layout(
                template="plotly_dark",
                paper_bgcolor='black',
                plot_bgcolor='black',
                xaxis_title="Predicted Price (USD)",
                yaxis_title="Frequency",
                bargap=0.1
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption(f"Based on {len(dist_df)} distinct simulation points recorded for this date.")
