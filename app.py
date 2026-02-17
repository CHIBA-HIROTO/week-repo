import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
import glob
import os
import requests
import google.generativeai as genai
import numpy as np
import json
import bcrypt

# --- 1. CONFIG & CSS ---
st.set_page_config(layout="wide", page_title="Urban Farming Dashboard")

# Custom CSS for Modern, White & Green Theme
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    /* Background */
    .stApp {
        background-color: #f4f6f8; /* Light gray background */
    }

    /* Cards */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2e7d32 !important; /* Green tone */
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #2e7d32;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #1b5e20;
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS (Preserved) ---

def get_ai_response(api_key, system_prompt, user_message, history=[]):
    """Gemini API (Unchanged)"""
    candidate_models = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
    genai.configure(api_key=api_key)
    last_error = ""
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            chat = model.start_chat(history=history)
            response = chat.send_message(f"{system_prompt}\n\nUser: {user_message}")
            return response.text
        except Exception as e:
            last_error = str(e)
            continue
    return f"AI Error: {last_error}"

def safe_read_csv(file_path_or_buffer):
    if file_path_or_buffer is None: return None
    try: return pd.read_csv(file_path_or_buffer, encoding='utf-8')
    except: return pd.read_csv(file_path_or_buffer, encoding='cp932')

def calculate_vpd(temp, humidity):
    if pd.isna(temp) or pd.isna(humidity): return 0
    e_sat = 6.1078 * 10**((7.5 * temp) / (temp + 237.3))
    return ((217 * e_sat) / (temp + 273.15)) * (100 - humidity) / 100

@st.cache_data(ttl=300)
def load_all_uecs_data(folder_path):
    all_files = glob.glob(f'{folder_path}/*.csv')
    if not all_files: return None
    df_list = []
    
    # Column mapping for Farm C compatibility (Standardize to Farm A/B format)
    col_map = {
        '内部気温[C]': '室内気温[C]',
        '内部相対湿度[%]': '室内相対湿度[%]',
        '内部CO2[ppm]': '室内CO2濃度[ppm]',
        '内部日射[kW m-2]': '室内日射強度[kW m-2]'
    }

    for file in all_files:
        df = safe_read_csv(file)
        if df is not None:
             # Normalize columns
             df = df.rename(columns=col_map)
             
             dt_col = df.columns[0]
             df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
             df = df.dropna(subset=[dt_col])
             df_list.append(df)
    if df_list:
        df_all = pd.concat(df_list)
        return df_all.sort_values(df_all.columns[0])
    return None

def parse_japanese_date(date_str):
    if pd.isna(date_str): return pd.NaT
    date_str = str(date_str).strip()
    try: return pd.to_datetime(date_str)
    except:
        try:
            if '月' in date_str and '日' in date_str:
                return pd.to_datetime(f"2025/{date_str.replace('月', '/').replace('日', '')}")
            return pd.to_datetime(date_str)
        except: return pd.NaT

# --- 3. STATE MANAGEMENT ---
# --- 3. AUTHENTICATION & STATE MANAGEMENT ---
AUTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auth.json')
# Master Password: "developer_master" (hashed for security in real app, but for now hardcoded hash of 'admin123' for demo?? User asked for developer access without constraints)
# Let's use a specific hash for "masterkey123"
# generated via bcrypt.hashpw(b"masterkey123", bcrypt.gensalt())
MASTER_PASSWORD_HASH = b'$2b$12$K.X.8.X.8.X.8.X.8.X.8.u.X.8.X.8.X.8.X.8.X.8.X.8.X.8' # Placeholder, will generate real one in code if needed or just use logic

def load_auth_data():
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, 'r') as f:
            return json.load(f)
    return {"users": {}}

def save_auth_data(data):
    with open(AUTH_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def init_auth_system():
    # Initial Migration: If no auth file, create from legacy hardcoded users
    if not os.path.exists(AUTH_FILE):
        # Legacy: USERS = {"カルナエスト": "aaaa", "SLOW FARM": "bbbb", "上原さん家のいちご園": "cccc"}
        # User folders map needs to be preserved
        initial_users = {
            "カルナエスト": {"password": "aaaa", "folder": "farm_a"},
            "SLOW FARM": {"password": "bbbb", "folder": "farm_b"},
            "上原さん家のいちご園": {"password": "cccc", "folder": "farm_c"}
        }
        
        auth_data = {"users": {}}
        for uid, info in initial_users.items():
            auth_data["users"][uid] = {
                "password_hash": hash_password(info["password"]),
                "data_folder": info["folder"]
            }
        save_auth_data(auth_data)

# Run initialization
init_auth_system()
auth_db = load_auth_data()

# Master Key Logic (Hardcoded hash for 'devpass999' for demonstration)
# Let's actually just check against a hardcoded string for simplicity in this specific request context 
# unless user wants strict security. "Developer can see without constraint".
# We will use a special check in login.
MASTER_KEY = "devpass999" 

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user_id'] = None
    st.session_state['is_master'] = False

# --- 4. LAYOUT IMPLEMENTATION ---

# --- 4. LAYOUT IMPLEMENTATION ---

# --- HEADER: LOGIN & SETTINGS ---
with st.container(border=True):
    col_header_left, col_header_right = st.columns([1, 1], gap="large")
    
    # Left: Login / User Info
    with col_header_left:
        st.markdown("### 🔐 User Login")
        if not st.session_state['authenticated']:
            c_l1, c_l2, c_l3 = st.columns([1, 1, 1])
            # Load users from DB
            user_list = list(auth_db['users'].keys()) if auth_db and 'users' in auth_db else []
            
            with c_l1: user_id = st.selectbox("Farm ID", user_list, label_visibility="collapsed")
            with c_l2: password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            with c_l3: 
                if st.button("Login", use_container_width=True):
                    # 1. Check Master Key
                    if password == MASTER_KEY:
                         st.session_state['authenticated'] = True
                         st.session_state['user_id'] = user_id
                         st.session_state['is_master'] = True
                         st.success(f"Login as Master (Viewing {user_id})")
                         st.rerun()
                    
                    # 2. Check User Password
                    elif user_id in auth_db['users']:
                        stored_hash = auth_db['users'][user_id]['password_hash']
                        if verify_password(password, stored_hash):
                            st.session_state['authenticated'] = True
                            st.session_state['user_id'] = user_id
                            st.session_state['is_master'] = False
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    else:
                        st.error("User not found")
        else:
            current_user = st.session_state['user_id']
            is_master = st.session_state.get('is_master', False)
            role_badge = "👑 Master" if is_master else "👤 User"
            
            c_u1, c_u2 = st.columns([3, 1])
            with c_u1: st.success(f"Welcome, **{current_user}** ({role_badge})")
            with c_u2: 
                if st.button("Logout", use_container_width=True):
                    st.session_state['authenticated'] = False
                    st.session_state['user_id'] = None
                    st.session_state['is_master'] = False
                    st.rerun()

    # Right: Settings
    with col_header_right:
        st.markdown("### ⚙️ Settings")
        c_s1, c_s2 = st.columns(2)
        with c_s1: lat = st.number_input("Latitude", value=35.44, format="%.2f")
        with c_s2: lon = st.number_input("Longitude", value=139.64, format="%.2f")
        
        # Change Password UI (Only if authenticated and NOT Master)
        if st.session_state['authenticated'] and not st.session_state.get('is_master', False):
            with st.expander("🔐 Change Password"):
                with st.form("change_pass_form"):
                    curr_pass = st.text_input("Current Password", type="password")
                    new_pass = st.text_input("New Password", type="password")
                    conf_pass = st.text_input("Confirm New Password", type="password")
                    
                    if st.form_submit_button("Update Password"):
                        u_data = auth_db['users'].get(st.session_state['user_id'])
                        if u_data and verify_password(curr_pass, u_data['password_hash']):
                            if new_pass == conf_pass and new_pass:
                                # Update
                                new_hash = hash_password(new_pass)
                                auth_db['users'][st.session_state['user_id']]['password_hash'] = new_hash
                                save_auth_data(auth_db)
                                st.success("Password updated successfully!")
                            else:
                                st.error("New passwords do not match or are empty.")
                        else:
                            st.error("Incorrect current password.")

# --- MAIN LOGIC (Only if Authenticated) ---
if st.session_state['authenticated']:
    current_user = st.session_state['user_id']
    # Get folder from Auth DB
    data_folder = "farm_a" # Default
    if current_user in auth_db['users']:
        data_folder = auth_db['users'][current_user].get('data_folder', 'farm_a')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, "data", data_folder)
    
    # --- UPPER SECTION: ENVIRONMENT DASHBOARD ---
    with st.container(border=True):
        st.markdown("### 🌤️ Environmental Monitoring")
        # Data Loading
        uecs_dir = f'{DATA_ROOT}/uecs'
        
        # 1. Scan for available files and parse weeks
        week_options = {} # Label -> Filename
        uecs_files = sorted(glob.glob(f'{uecs_dir}/*.csv'))
        
        for f_path in uecs_files:
            try:
                basename = os.path.basename(f_path)
                # Expected format: YYYYMMDDYYYYMMDD_ka.csv (StartEnd_ka.csv)
                name_part = basename.split('_')[0]
                if len(name_part) == 12: # YYYYMMDDMMDD
                    year = name_part[:4]
                    m1, d1 = name_part[4:6], name_part[6:8]
                    label = f"{year}/{m1}/{d1} - {name_part[8:10]}/{name_part[10:12]}"
                    week_options[label] = f_path
                else:
                    week_options[basename] = f_path
            except:
                week_options[os.path.basename(f_path)] = f_path

        # Load ALL data for "Latest" or "Custom" mode
        df_u = load_all_uecs_data(uecs_dir)
        
        if df_u is not None:
            dt_col = df_u.columns[0]
            min_date_all = df_u[dt_col].min().date()
            max_date_all = df_u[dt_col].max().date()
        else:
            st.warning("No valid environmental data found in any files.")
            dt_col = None
        
        # Controls
        c1, c2 = st.columns([1, 1])
        with c1:
            # Mode Selection
            display_mode = st.radio("Display Mode", ["Latest Week", "Select Week", "Custom Range"], horizontal=True)
            
            date_range = []
            selected_week_file = None
            
            if display_mode == "Latest Week":
                if df_u is not None:
                    # Default to last 7 days of ALL DATA
                    default_end = max_date_all
                    default_start = default_end - timedelta(days=6)
                    if default_start < min_date_all: default_start = min_date_all
                    date_range = [default_start, default_end]
                    st.caption(f"Showing: {default_start} - {default_end}")
                else:
                    st.caption("No data to display.")

            elif display_mode == "Select Week":
                # Show dataframe from SPECIFIC FILE
                if week_options:
                    selected_label = st.selectbox("Choose Week", list(week_options.keys()), index=len(week_options)-1)
                    if selected_label:
                        selected_week_file = week_options[selected_label]
                else:
                    st.warning("No weekly files found.")
            
            elif display_mode == "Custom Range":
                if df_u is not None:
                    date_range = st.slider(
                        "Display Period",
                        min_value=min_date_all,
                        max_value=max_date_all,
                        value=(min_date_all, max_date_all)
                    )
        
        with c2:
            # Metric Selection
            metric_options = {
                '室内気温[C]': {'label': 'Temperature', 'unit': '℃', 'color': '#d32f2f'},
                '室内相対湿度[%]': {'label': 'Humidity', 'unit': '%', 'color': '#1976d2'},
                '室内CO2濃度[ppm]': {'label': 'CO2', 'unit': 'ppm', 'color': '#388e3c'},
                '室内日射強度[kW m-2]': {'label': 'Solar', 'unit': 'kW/m²', 'color': '#fbc02d'},
                '積算日射[MJ]': {'label': 'Cumulative Solar', 'unit': 'MJ', 'color': '#f57c00'},
                '飽差': {'label': 'VPD', 'unit': 'g/m³', 'color': '#7b1fa2'}
            }
            
            if df_u is not None:
                if '飽差' not in df_u.columns and '室内気温[C]' in df_u.columns and '室内相対湿度[%]' in df_u.columns:
                        df_u['飽差'] = df_u.apply(lambda x: calculate_vpd(x['室内気温[C]'], x['室内相対湿度[%]']), axis=1)

            available_metrics = list(metric_options.keys())
            if df_u is not None:
                    available_metrics = [c for c in metric_options.keys() if c in df_u.columns]
            
            selected_metrics = st.multiselect("Metrics", available_metrics, default=[m for m in available_metrics if '気温' in m or '湿度' in m])

        # Prepare filtered DataFrame based on Mode
        df_filtered = pd.DataFrame()
        
        if display_mode == "Select Week" and selected_week_file:
            # Load SPECIFIC file only
            try:
                df_week = safe_read_csv(selected_week_file)
                if df_week is not None and not df_week.empty:
                        # Preprocess standard
                        # Normalization for Farm C
                        col_map_week = {
                            '内部気温[C]': '室内気温[C]',
                            '内部相対湿度[%]': '室内相対湿度[%]',
                            '内部CO2[ppm]': '室内CO2濃度[ppm]',
                            '内部日射[kW m-2]': '室内日射強度[kW m-2]'
                        }
                        df_week = df_week.rename(columns=col_map_week)

                        w_dt_col = df_week.columns[0]
                        df_week[w_dt_col] = pd.to_datetime(df_week[w_dt_col], errors='coerce')
                        df_week = df_week.dropna(subset=[w_dt_col]).sort_values(w_dt_col)
                        
                        if '飽差' not in df_week.columns and '室内気温[C]' in df_week.columns and '室内相対湿度[%]' in df_week.columns:
                            df_week['飽差'] = df_week.apply(lambda x: calculate_vpd(x['室内気温[C]'], x['室内相対湿度[%]']), axis=1)
                        
                        df_filtered = df_week
                        dt_col = w_dt_col
                else:
                        st.warning(f"⚠️ Selected file ({os.path.basename(selected_week_file)}) appears to be empty.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        elif df_u is not None and date_range:
            # Filter global dataframe
            mask = (df_u[dt_col].dt.date >= date_range[0]) & (df_u[dt_col].dt.date <= date_range[1])
            df_filtered = df_u.loc[mask].copy()

        # --- Daily Summary Section (New Feature) ---
        if not df_filtered.empty:
            if '室内気温[C]' in df_filtered.columns:
                st.markdown("##### 📅 Weekly Temperature Summary")
                # Resample by day
                df_daily = df_filtered.set_index(dt_col).resample('D')['室内気温[C]'].agg(['max', 'min', 'mean']).reset_index()
                
                # Create 7 columns (or fewer if less data)
                cols = st.columns(7)
                for idx, row in df_daily.iterrows():
                    if idx < 7: # Limit to 7 columns to avoid overflow if range > 1 week
                        date_str = row[dt_col].strftime('%m/%d (%a)')
                        with cols[idx]:
                            st.caption(f"**{date_str}**")
                            st.markdown(f"<span style='color:#d32f2f'>Max: {row['max']:.2f}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span style='color:#388e3c'>Avg: {row['mean']:.2f}</span>", unsafe_allow_html=True) # Light Green used #388e3c (Material Green 700) for better visibility on white
                            st.markdown(f"<span style='color:#1976d2'>Min: {row['min']:.2f}</span>", unsafe_allow_html=True)

        # Plotting
        if not df_filtered.empty and selected_metrics:
            fig = go.Figure()
            layout_args = dict(
                height=400,
                template="plotly_white",
                hovermode="x unified",
                xaxis=dict(
                    domain=[0.1 + (len(selected_metrics)-1)*0.05, 1.0],
                    rangeslider=dict(visible=True),
                    type="date",
                    showgrid=True
                ),
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                margin=dict(l=20, r=20, t=50, b=50)
            )

            for i, col in enumerate(selected_metrics):
                if col in df_filtered.columns:
                    meta = metric_options.get(col, {'label': col, 'unit': '', 'color': 'gray'})
                    y_axis_name = f"yaxis{i+1}" if i > 0 else "yaxis"
                    
                    fig.add_trace(go.Scatter(
                        x=df_filtered[dt_col],
                        y=df_filtered[col],
                        name=f"{meta['label']} ({meta['unit']})",
                        line=dict(width=2, color=meta['color']),
                        yaxis=f"y{i+1}" if i > 0 else "y",
                        hovertemplate='%{y:.2f}'
                    ))
                    
                    side = "left" if i == 0 else "right"
                    
                    layout_args[y_axis_name] = dict(
                        title=dict(text=meta['unit'], font=dict(color=meta['color'])),
                        tickfont=dict(color=meta['color']),
                        anchor="x",
                        overlaying="y" if i > 0 else None,
                        side=side,
                        showgrid=(i==0)
                    )
                    if i > 0:
                            layout_args[y_axis_name]['position'] = 1.0 if i==1 else 1.0 - ((i-1) * 0.06)
                            layout_args[y_axis_name]['anchor'] = 'free'

            fig.update_layout(**layout_args)
            if len(selected_metrics) > 1:
                fig.update_layout(xaxis=dict(domain=[0, 1.0 - (len(selected_metrics)-1)*0.06]))

            st.plotly_chart(fig, use_container_width=True)
        elif display_mode != "Select Week" or (display_mode == "Select Week" and selected_week_file): 
                if df_filtered.empty and df_u is not None:
                    st.info("No data selected or available for this range.")

    # --- LOWER SECTION: GROWTH DATA ---
    with st.container(border=True):
        st.markdown("### 🌱 Growth Progress")
        
        growth_files = glob.glob(f'{DATA_ROOT}/growth/*.csv')
        if growth_files:
             # Load Growth Data with FILENAME-BASED DATE PARSING
            all_g_data = []
            new_cols = ['date', 'area', 'no', '草高(cm)', '第3葉柄長(cm)', '葉数', '蕾数', '花数', '肥大果数', '緑熟果数', '白熟果数']
            calc_cols = ['草高(cm)', '第3葉柄長(cm)', '葉数', '蕾数', '花数', '肥大果数', '緑熟果数', '白熟果数'] # Added '花数'
            
            for f_path in growth_files:
                try:
                    # Extract date from filename: "YYYYMMDD_ka.csv"
                    basename = os.path.basename(f_path)
                    date_part = basename.split('_')[0] # "YYYYMMDD"
                    file_date = pd.to_datetime(date_part, format='%Y%m%d')
                    
                    df_tmp = safe_read_csv(f_path)
                    if df_tmp is not None and len(df_tmp.columns) >= len(new_cols):
                        df_tmp = df_tmp.iloc[:, :len(new_cols)]
                        df_tmp.columns = new_cols
                        # OVERWRITE date column with the one from filename
                        df_tmp['date'] = file_date
                        all_g_data.append(df_tmp)
                except Exception as e:
                    # Fallback or skip
                    # print(f"Error loading {f_path}: {e}")
                    pass
            
            if all_g_data:
                df_growth = pd.concat(all_g_data).dropna(subset=['date']).sort_values('date')
                for c in calc_cols: df_growth[c] = pd.to_numeric(df_growth[c], errors='coerce')

                # KPI Cards (Latest Average)
                latest_date = df_growth['date'].max()
                df_latest = df_growth[df_growth['date'] == latest_date]
                avg_latest = df_latest[calc_cols].mean()
                
                st.caption(f"Latest Survey: {latest_date.strftime('%Y/%m/%d')}")
                kpi_cols = st.columns(len(calc_cols[:5])) # Show top 5 metrics (Height, Petiole, Leaves, Buds, Flowers)
                for i, col in enumerate(calc_cols[:5]):
                    with kpi_cols[i]:
                        st.metric(label=col, value=f"{avg_latest[col]:.2f}")
                
                # Growth Graphs - Split into 3 Sections
                st.write("#### Growth Trend Analysis")
                df_trend = df_growth.groupby('date')[calc_cols].mean().reset_index()
                
                # Calculate Total Fruit Count (着果数)
                # Assuming cols exist: 肥大果数, 緑熟果数, 白熟果数
                fruit_cols = ['肥大果数', '緑熟果数', '白熟果数']
                # Ensure they are numeric
                for c in fruit_cols:
                    if c not in df_trend.columns: df_trend[c] = 0
                
                df_trend['着果数'] = df_trend[fruit_cols].sum(axis=1)

                # 1. Top: Morphology (Line) - Height, Petiole
                st.caption("1. Morphology (Height, Petiole)")
                fig_morph = go.Figure()
                for c in ['草高(cm)', '第3葉柄長(cm)']:
                    if c in df_trend.columns:
                        fig_morph.add_trace(go.Scatter(x=df_trend['date'], y=df_trend[c], name=c, mode='lines+markers', hovertemplate='%{y:.2f}'))
                fig_morph.update_layout(height=300, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_morph, use_container_width=True)

                # 2. Middle: Counts (Line) - Leaves, Buds, Flowers, Total Fruits
                st.caption("2. Plant Organ Counts")
                fig_count = go.Figure()
                # '着果数' is now available
                count_targets = ['葉数', '蕾数', '花数', '着果数'] 
                colors = {'葉数': '#2e7d32', '蕾数': '#f9a825', '花数': '#d81b60', '着果数': '#c62828'}
                
                for c in count_targets:
                    if c in df_trend.columns:
                        fig_count.add_trace(go.Scatter(x=df_trend['date'], y=df_trend[c], name=c, mode='lines+markers', line=dict(color=colors.get(c)), hovertemplate='%{y:.2f}'))
                fig_count.update_layout(height=300, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_count, use_container_width=True)

                # 3. Bottom: Fruit Details (Bar) - Enlarged, Green, White
                st.caption("3. Fruit Composition")
                fig_fruit = go.Figure()
                fruit_colors = {'肥大果数': '#d32f2f', '緑熟果数': '#388e3c', '白熟果数': '#fbc02d'} # Red, Green, Yellow-ish
                
                for c in fruit_cols:
                    if c in df_trend.columns:
                        fig_fruit.add_trace(go.Bar(x=df_trend['date'], y=df_trend[c], name=c, marker_color=fruit_colors.get(c), hovertemplate='%{y:.2f}'))
                
                # Custom X-axis Labels with Total Fruit Count
                tick_vals = df_trend['date']
                tick_text = [f"{d.strftime('%m/%d')}<br>Total: {t:.2f}" for d, t in zip(df_trend['date'], df_trend['着果数'])]
                
                fig_fruit.update_layout(
                    height=300, 
                    template="plotly_white", 
                    barmode='group',
                    margin=dict(l=20, r=20, t=20, b=50), # Increased bottom margin for labels
                    legend=dict(orientation="h", y=1.1),
                    yaxis_title="Fruit Count",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=tick_vals,
                        ticktext=tick_text
                    )
                )
                st.plotly_chart(fig_fruit, use_container_width=True)
            else:
                st.warning("No valid growth data loaded.")
        else:
             st.info("No growth CSV files found.")

    # --- NEW SECTION: YIELD PREDICTION SIMULATION ---
    with st.container(border=True):
        st.markdown("### 🔮 Yield Prediction Simulation")
        st.caption("Calculate potential yield based on physiological parameters and environmental conditions.")
        
        # 1. Layout: Parameters (Sidebar-like) and Scenario/Results
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.markdown("#### 🛠 Parameters")
            # Biological Constants
            planting_density = st.number_input("Planting Density (plants/m²)", value=2.5, step=0.1, help="栽植密度")
            leaf_area_cm2 = st.number_input("Indiv. Leaf Area (cm²)", value=500.0, step=10.0, help="個葉面積 (推定)")
            extinction_coeff = st.number_input("Extinction Coeff (k)", value=0.7, step=0.01, help="吸光係数")
            lue = st.number_input("LUE (g/MJ)", value=3.0, step=0.1, help="光利用効率 (Light Use Efficiency)")
            partitioning_rate = st.number_input("Fruit Partitioning", value=0.6, step=0.05, max_value=1.0, help="果実分配率")
            dry_matter_content = st.number_input("Fruit Dry Matter %", value=0.08, step=0.01, max_value=1.0, help="果実乾物率")

        with sim_col2:
            st.markdown("#### 🌤 Historical Analysis")
            
            # --- INPUTS for Historical Mode ---
            hist_c1, hist_c2, hist_c3 = st.columns(3)
            with hist_c1:
                # Default planting date: 3 months ago or first data date
                default_plant_date = datetime.now().date() - timedelta(days=90)
                if 'df_u' in locals() and df_u is not None:
                     default_plant_date = df_u[dt_col].min().date()

                planting_date = st.date_input("Planting Date", value=default_plant_date, help="定植日 (積算気温の開始日)")
            
            with hist_c2:
                # Location (from Settings)
                st.markdown(f"**Location**")
                st.caption(f"Lat: {lat} / Lon: {lon}")
                st.caption("(Change in Top Settings)")
            
            with hist_c3:
                # Offsets
                temp_offset = st.number_input("Temp Offset (+℃)", value=3.0, step=0.5, help="外気温に対するハウス内昇温目安")
                solar_transmissivity = st.number_input("Solar Transmissivity", value=0.7, step=0.05, max_value=1.0, help="ハウス内日射透過率")

            # --- RUN SIMULATION ---
            if st.button("Run Simulation (Historical Data)", type="primary"):
                with st.spinner("Fetching weather data and simulating..."):
                    # 1. Prepare Date Range
                    today = datetime.now().date()
                    if planting_date >= today:
                        st.error("Planting date must be in the past.")
                    else:
                        # 2. Fetch Open-Meteo Data (Outside)
                        try:
                            url = "https://archive-api.open-meteo.com/v1/archive"
                            params = {
                                "latitude": lat,
                                "longitude": lon,
                                "start_date": planting_date.strftime("%Y-%m-%d"),
                                "end_date": today.strftime("%Y-%m-%d"),
                                "daily": "temperature_2m_mean,shortwave_radiation_sum",
                                "timezone": "Asia/Tokyo"
                            }
                            res = requests.get(url, params=params)
                            res.raise_for_status()
                            data = res.json()
                            
                            df_om = pd.DataFrame({
                                'date': pd.to_datetime(data['daily']['time']).date,
                                'temp_out': data['daily']['temperature_2m_mean'],
                                'solar_out_mj': data['daily']['shortwave_radiation_sum']
                            })
                            # Estimate Inside Conditions
                            df_om['temp_est'] = df_om['temp_out'] + temp_offset
                            df_om['solar_est'] = df_om['solar_out_mj'] * solar_transmissivity
                            df_om = df_om.set_index('date')
                            
                        except Exception as e:
                            st.error(f"Error fetching Open-Meteo data: {e}")
                            st.stop()
                        
                        # 3. Prepare Actual UECS Data (Daily Mean)
                        df_uecs_daily = pd.DataFrame()
                        if 'df_u' in locals() and df_u is not None:
                            # Ensure numeric types before resampling
                            cols_to_resample = ['室内気温[C]', '室内日射強度[kW m-2]', '室内CO2濃度[ppm]']
                            # Check existence
                            exist_cols = [c for c in cols_to_resample if c in df_u.columns]
                            if exist_cols:
                                for c in exist_cols:
                                    df_u[c] = pd.to_numeric(df_u[c], errors='coerce')
                                
                                df_uecs_daily = df_u.set_index(dt_col).resample('D')[exist_cols].mean()
                                df_uecs_daily.index = df_uecs_daily.index.date
                                
                                # Convert kW/m2 (instant) to MJ/m2/day ?
                                # UECS '室内日射強度' is usually instantaneous kW/m2. Mean * 24h * 3.6?
                                # Wait, '積算日射[MJ]' might be better if available.
                                # Let's assume Mean kW/m2 * (Sunshine Hours?) -> Gross estimate: Mean(kW) * 24 * 3.6 = MJ/day?
                                # Actually, Solar Radiation is 0 at night. Mean includes zeros. 
                                # So Mean(kW/m2 over 24h) * 86400 sec / 1000000 = MJ/m2.
                                # Mean(kW) = kJ/s.  Mean * 86400 = kJ/day. / 1000 = MJ/day.
                                if '室内日射強度[kW m-2]' in df_uecs_daily.columns:
                                    df_uecs_daily['solar_mj'] = df_uecs_daily['室内日射強度[kW m-2]'] * 86.4
                                
                        # 4. Merge Environmental Data
                        # Index: planting_date to today
                        full_dates = pd.date_range(planting_date, today).date
                        df_sim = pd.DataFrame(index=full_dates)
                        
                        # Join External (Estimated)
                        df_sim = df_sim.join(df_om[['temp_est', 'solar_est']], how='left')
                        
                        # Join Actual (UECS) - Overwrite Estimated where available
                        if not df_uecs_daily.empty:
                            df_sim = df_sim.join(df_uecs_daily, how='left')
                            
                            # Create final columns combining Est and Act
                            # Prefer Act ('室内気温[C]'), fallback to Est ('temp_est')
                            df_sim['temp_final'] = df_sim['室内気温[C]'].combine_first(df_sim['temp_est'])
                            
                            if 'solar_mj' in df_sim.columns:
                                df_sim['solar_final'] = df_sim['solar_mj'].combine_first(df_sim['solar_est'])
                            else:
                                df_sim['solar_final'] = df_sim['solar_est']
                            
                            # CO2: Actual or Default 400
                            if '室内CO2濃度[ppm]' in df_sim.columns:
                                df_sim['co2_final'] = df_sim['室内CO2濃度[ppm]'].fillna(400)
                            else:
                                df_sim['co2_final'] = 400
                        else:
                            df_sim['temp_final'] = df_sim['temp_est']
                            df_sim['solar_final'] = df_sim['solar_est']
                            df_sim['co2_final'] = 400
                            
                        # Fill any remaining NaNs (e.g. today's future hours?)
                        df_sim = df_sim.ffill().bfill()
                        
                        # 5. Prepare Growth Data (Leaf Count Interpolation)
                        # Create daily leaf count series
                        # Default curve: 0 at planting -> interpolated -> last value
                        
                        raw_growth = []
                        # Add start point (Planting Date, 0 leaves)
                        raw_growth.append({'date': planting_date, 'leaves': 0.0})
                        
                        if 'df_growth' in locals() and not df_growth.empty:
                            # Use actual surveys
                            # Filter surveys after planting date
                            valid_surveys = df_growth[df_growth['date'].dt.date >= planting_date].sort_values('date')
                            for _, row in valid_surveys.iterrows():
                                if not pd.isna(row.get('葉数')):
                                    raw_growth.append({'date': row['date'].date(), 'leaves': float(row['葉数'])})
                        
                        df_leaves_raw = pd.DataFrame(raw_growth).drop_duplicates('date').set_index('date')
                        
                        # Reindex to full simulation range and interpolate
                        df_sim = df_sim.join(df_leaves_raw, how='left')
                        df_sim['leaves_interp'] = df_sim['leaves'].interpolate(method='linear').fillna(0) # Fillna 0 for initial
                        # If latest survey is old, extend the last known value
                        df_sim['leaves_interp'] = df_sim['leaves_interp'].ffill()

                        # --- 6. SIMULATION LOOP ---
                        # Variables
                        cum_gdd = 0.0
                        cum_yield = 0.0
                        
                        results = []
                        
                        for date, row in df_sim.iterrows():
                            # GDD (Base 10)
                            t_avg = row['temp_final']
                            gdd = max(0.0, t_avg - 10.0)
                            cum_gdd += gdd
                            
                            # Yield
                            leaves = row['leaves_interp']
                            solar = row['solar_final']
                            
                            # Logic
                            lai = leaves * (leaf_area_cm2 / 10000.0) * planting_density
                            interception_rate = 1.0 - math.exp(-extinction_coeff * lai)
                            intercepted_light = solar * interception_rate
                            dry_matter = intercepted_light * lue
                            dry_fruit = dry_matter * partitioning_rate
                            fresh_yield_daily = dry_fruit / dry_matter_content # g/m2 = kg/10a equivalent logic used before
                            
                            cum_yield += fresh_yield_daily
                            
                            results.append({
                                'date': date,
                                'GDD': cum_gdd,
                                'Yield_Daily': fresh_yield_daily,
                                'Yield_Cum': cum_yield,
                                'LAI': lai
                            })
                            
                        df_res = pd.DataFrame(results)
                        
                        # --- 7. VISUALIZATION ---
                        st.write(f"**Predicted Total Yield: {cum_yield:.2f} kg/10a** (GDD: {cum_gdd:.1f})")
                        
                        # Dual Axis Graph
                        fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Yield Line
                        fig_sim.add_trace(go.Scatter(
                            x=df_res['date'], y=df_res['Yield_Cum'],
                            name="Cum. Yield (kg/10a)",
                            line=dict(color='#2E7D32', width=3),
                            hovertemplate='%{y:.2f} kg/10a'
                        ), secondary_y=False)
                        
                        # GDD Area
                        fig_sim.add_trace(go.Scatter(
                            x=df_res['date'], y=df_res['GDD'],
                            name="Accumulated GDD",
                            fill='tozeroy',
                            line=dict(color='rgba(255, 160, 0, 0.5)', width=0),
                            fillcolor='rgba(255, 160, 0, 0.1)',
                            hovertemplate='%{y:.1f} ℃'
                        ), secondary_y=True)
                        
                        fig_sim.update_layout(
                            height=400,
                            title="Yield Prediction vs Accumulated Temperature",
                            template="plotly_white",
                            xaxis_title="Date",
                            hovermode="x unified",
                            legend=dict(orientation="h", y=1.1)
                        )
                        fig_sim.update_yaxes(title_text="Yield (kg/10a)", secondary_y=False)
                        fig_sim.update_yaxes(title_text="GDD (℃)", secondary_y=True)
                        
                        st.plotly_chart(fig_sim, use_container_width=True)


else:
    # Non-authenticated landing (filled by Left Column's content, main area empty)
    st.info("👆 Please authorize from the top panel.")
