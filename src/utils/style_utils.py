import streamlit as st

def inject_industrial_theme():
    """Injects high-performance, lightweight CSS for the Tactical HUD aesthetic."""
    st.markdown(
        """
        <style>
        /* 1. Global HUD Aesthetics */
        .stApp {
            background-color: #0c1117;
            color: #e6edf3;
        }
        
        /* 2. Glassmorphism Metric Cards */
        [data-testid="stMetricValue"] {
            color: #00ffff !important;
            font-family: 'Courier New', Courier, monospace;
            font-weight: bold;
        }
        
        [data-testid="stMetricLabel"] {
            color: #8b949e !important;
            font-family: 'Verdana', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.1rem;
            font-size: 0.7rem !important;
        }
        
        /* Tactical Card Housing */
        .hud-card {
            background: rgba(22, 27, 34, 0.7);
            border: 1px solid rgba(0, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }
        
        /* Status Badges */
        .status-badge {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .status-success {
            background: rgba(0, 255, 255, 0.1);
            color: #00ffff;
            border: 1px solid #00ffff;
        }
        
        .status-warning {
            background: rgba(255, 153, 0, 0.1);
            color: #ff9900;
            border: 1px solid #ff9900;
        }
        
        /* Sidebar Refinement */
        [data-testid="stSidebar"] {
            background-color: #010409;
            border-right: 1px solid rgba(0, 255, 255, 0.1);
        }
        
        /* Buttons Industrial Look */
        .stButton button {
            background-color: transparent !important;
            border: 1px solid rgba(0, 255, 255, 0.3) !important;
            color: #00ffff !important;
            transition: all 0.2s ease-in-out;
        }
        
        .stButton button:hover {
            border: 1px solid #00ffff !important;
            background-color: rgba(0, 255, 255, 0.1) !important;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
        }
        
        /* Chart Optimization */
        .main-chart-container {
            border: 1px solid rgba(0, 255, 255, 0.1);
            border-radius: 12px;
            padding: 5px;
            background: #0d1117;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_glass_card(content: str, title: str = None):
    """Wraps content in a tactical glass container."""
    if title:
        st.markdown(f"<div class='hud-card'><div style='color: grey; font-size: 0.7rem; margin-bottom: 0.5rem;'>{title.upper()}</div>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='hud-card'>{content}</div>", unsafe_allow_html=True)
