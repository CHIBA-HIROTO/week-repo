import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
import glob
import os
import requests
import google.generativeai as genai

# --- 1. 計算・ユーティリティロジック ---

def get_ai_response(api_key, system_prompt, user_message, history=[]):
    """Gemini APIを呼び出して回答を取得 (複数モデルでフォールバック試行)"""
    # 試行するモデルの候補リスト (ユーザー指定の2.5, 最新の1.5, 安定版1.5, 旧Pro)
    candidate_models = [
        'gemini-2.5-flash',       # ユーザー示唆/最新検索結果
        'gemini-1.5-flash',       # 標準エイリアス
        'gemini-1.5-flash-latest',# 最新エイリアス
        'gemini-1.5-flash-001',   # 固定バージョン
        'gemini-1.5-pro',         # Pro版 (無料枠範囲内で利用可の場合)
        'gemini-pro'              # 旧安定版
    ]
    
    genai.configure(api_key=api_key)
    last_error = ""

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            chat = model.start_chat(history=history)
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"
            response = chat.send_message(full_prompt)
            return response.text # 成功したら即座に返す
        except Exception as e:
            last_error = str(e)
            # 404 (Not Found) 以外なら即座にエラーとして扱うべきかもしれないが、
            # 今回はモデル名起因の可能性が高いため次を試行する
            continue

    return f"AI回答エラー (全てのモデルで失敗): {last_error}"

def fetch_weather_data(lat, lon, start_date=None, end_date=None):
    """Open-Meteo APIから天気データを取得 (過去・現在・未来)"""
    try:
        # 基本URL (Forecast API)
        url = "https://api.open-meteo.com/v1/forecast"
        
        # パラメータ設定
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "Asia/Tokyo"
        }
        
        # 期間指定がある場合 (過去データの取得用)
        if start_date and end_date:
            # 日付型を文字列(YYYY-MM-DD)に変換
            params["start_date"] = start_date.strftime('%Y-%m-%d')
            params["end_date"] = end_date.strftime('%Y-%m-%d')
            # 過去データを含む場合もForecast APIで92日前までは取得可能
            # それ以上古い場合はarchive APIが必要だが、今回は簡易的にforecastを使用
        else:
            # 指定なし＝週間予報モード (7日間)
            params["forecast_days"] = 7

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # DataFrameに変換
        hourly = data.get('hourly', {})
        if not hourly: return None
        
        df_weather = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'temp_out': hourly['temperature_2m'],
            'humid_out': hourly['relative_humidity_2m'],
            'weather_code': hourly['weather_code'],
            'wind_speed': hourly['wind_speed_10m']
        })
        return df_weather
        
    except Exception as e:
        st.error(f"天気データの取得に失敗しました: {e}")
        return None

def get_weather_icon(code):
    """WMO天気コードをアイコン/テキストに変換"""
    # 簡易マッピング
    if code == 0: return "☀️ 快晴"
    if 1 <= code <= 3: return "🌤️ 晴れ/曇り"
    if 45 <= code <= 48: return "🌫️ 霧"
    if 51 <= code <= 67: return "☂️ 雨"
    if 71 <= code <= 77: return "☃️ 雪"
    if 80 <= code <= 82: return "☔ にわか雨"
    if 95 <= code <= 99: return "⚡ 雷雨"
    return f"❓ 不明({code})"

def safe_read_csv(file_path_or_buffer):
    """文字コードエラーを回避してCSVを読み込む (パスorバッファ対応)"""
    if file_path_or_buffer is None: return None
    if isinstance(file_path_or_buffer, str):
        try: return pd.read_csv(file_path_or_buffer, encoding='utf-8')
        except: return pd.read_csv(file_path_or_buffer, encoding='cp932')
    else:
        file_path_or_buffer.seek(0)
        try: return pd.read_csv(file_path_or_buffer, encoding='utf-8')
        except:
            file_path_or_buffer.seek(0)
            return pd.read_csv(file_path_or_buffer, encoding='cp932')

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

def calculate_vpd(temp, humidity):
    if pd.isna(temp) or pd.isna(humidity): return 0
    e_sat = 6.1078 * 10**((7.5 * temp) / (temp + 237.3))
    return ((217 * e_sat) / (temp + 273.15)) * (100 - humidity) / 100

def get_sun_times(date, lat, lon):
    n = date.timetuple().tm_yday
    delta = 0.409 * math.sin(2 * math.pi * (n - 81) / 365)
    phi = math.radians(lat)
    try: h = math.acos(-math.tan(phi) * math.tan(delta))
    except: h = 0
    h_deg = math.degrees(h)
    lon_corr = (135 - lon) * 4 / 60
    return 12 + lon_corr - (h_deg / 15), 12 + lon_corr + (h_deg / 15)

# --- 1. 計算・ユーティリティロジック ---

# 簡易ユーザー管理 (デモ用: 本番では secrets.toml 推奨)
USERS = {
    "カルナエスト": "aaaa",
    "SLOW FARM": "bbbb",
    "上原さん家のいちご園": "cccc"
}

# フォルダとの紐付け
USER_FOLDERS = {
    "カルナエスト": "farm_a",
    "SLOW FARM": "farm_b",
    "上原さん家のいちご園": "farm_c"
}

def check_password():
    """パスワード認証 (サイドバー)"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None

    if st.session_state['authenticated']:
        return True

    st.sidebar.title("ログイン")
    user_id = st.sidebar.selectbox("農家IDを選択", list(USERS.keys()))
    password = st.sidebar.text_input("パスワード", type="password")
    
    if st.sidebar.button("ログイン"):
        if USERS[user_id] == password:
            st.session_state['authenticated'] = True
            st.session_state['user_id'] = user_id
            st.rerun()
        else:
            st.sidebar.error("パスワードが違います")
    return False



# --- 2. UI構成 ---
st.set_page_config(layout="wide", page_title="栽培支援レポート（アグリサイエンス研究室）")

# カスタムCSS (紫紺 & ライムグリーン: Modern Dashboard Style)
st.markdown("""
<style>
    /* フォント読み込み (Google Fonts: Noto Sans JP) */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif;
    }

    /* 変数定義 */
    :root {
        --primary-bg: #460e44; /* 紫紺 background */
        --primary-text: #ffffff;
        --accent-lime: #bfff00; /* ライムグリーン */
        --main-bg: #f8f9fa; /* メイン背景 (薄いグレー) */
    }

    /* --- サイドバー周り (紫紺ベース) --- */
    [data-testid="stSidebar"] {
        background-color: var(--primary-bg);
    }
    
    /* サイドバー内のテキスト全般を白に */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--primary-text) !important;
    }
    
    /* サイドバー内のWidgetラベル (入力フォームの上にある文字) */
    [data-testid="stSidebar"] label {
        color: var(--accent-lime) !important;
        font-weight: bold;
    }
    
    /* ※重要: 入力ボックス内の文字色は黒に戻す (そうしないと背景白で文字白になる) */
    [data-testid="stSidebar"] input, 
    [data-testid="stSidebar"] select, 
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #333 !important;
    }
    
    /* --- メインエリア --- */
    /* ヘッダー: ミニマルでモダンに */
    h1 {
        font-weight: 700;
        color: #333 !important;
        border-bottom: 2px solid var(--primary-bg);
        padding-bottom: 15px;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: var(--primary-bg) !important;
        font-weight: 600;
    }
    
    /* --- タブデザイン (シンプル・ミニマル) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #eee;
    }
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 0;
        padding: 5px 10px;
        border: none;
        background-color: transparentbox;
        transition: color 0.2s;
        margin-bottom: -1px; /* 下線と重ねる */
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 4px solid var(--accent-lime) !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
        color: var(--primary-bg) !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] p {
        color: #777;
    }
    /* タブホバー時の挙動 */
    .stTabs [data-baseweb="tab-list"] button:hover p {
        color: var(--primary-bg);
    }

    /* --- ボタン --- */
    /* Primaryボタン (分析ボタンなど) */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-bg), #6a1b66);
        color: var(--accent-lime) !important;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# タイトル表示
st.title("栽培支援レポート（アグリサイエンス研究室）")

# 認証チェック
if not check_password():
    st.stop()

# 認証済みユーザーのID
current_user = st.session_state['user_id']
# 表示名から実際のフォルダ名を解決
data_folder = USER_FOLDERS.get(current_user, "farm_a")
DATA_ROOT = f"data/{data_folder}"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.success(f"ログイン中: {current_user}")
    if st.button("ログアウト"):
        st.session_state['authenticated'] = False
        st.session_state['user_id'] = None
        st.rerun()
        
    st.header("データ設定")
    st.subheader("1. 環境データ(UECS)")
    # ユーザー固有パスから検索
    uecs_files = glob.glob(f'{DATA_ROOT}/uecs/*.csv')
    uecs_file_path = None
    if uecs_files:
        uecs_files.sort(key=os.path.getmtime, reverse=True)
        uecs_file_path = st.selectbox("読み込むファイルを選択", uecs_files, format_func=lambda x: os.path.basename(x))
        st.success(f"読み込み中: {os.path.basename(uecs_file_path)}")
    else: st.warning(f"{DATA_ROOT}/uecs/ フォルダにCSVファイルがありません。")

    st.subheader("2. 生育調査データ")
    growth_files = glob.glob(f'{DATA_ROOT}/growth/*.csv')
    if growth_files: st.info(f"{len(growth_files)} 個のファイルを検出しました。")
    else: st.warning(f"{DATA_ROOT}/growth/ フォルダにCSVファイルがありません。")
    
    st.divider()
    st.subheader("地点設定 (天気連携用)")
    lat = st.number_input("栽培地の緯度", value=35.44, format="%.2f")
    lon = st.number_input("栽培地の経度", value=139.64, format="%.2f")

    st.divider()
    st.subheader("AI設定 (Gemini)")
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studioで作成したAPIキーを入力してください")

# データ読み込み処理
df_u = safe_read_csv(uecs_file_path)

tab1, tab2, tab3, tab4 = st.tabs(["①生長推移データ", "②環境推移データ", "③天気予報 (New)", "④AI Agronomist (Beta)"])

# --- TAB 1: 生育レポート (省略 - 変更なし) ---
with tab1:
    if growth_files:
        all_data = []
        new_cols = ['date', 'area', 'no', '草高(cm)', '第3葉柄長(cm)', '葉数', '蕾数', '花数', '肥大果数', '緑熟果数', '白熟果数']
        calc_cols = ['草高(cm)', '第3葉柄長(cm)', '葉数', '蕾数', '花数', '肥大果数', '緑熟果数', '白熟果数']
        for f_path in growth_files:
            df_tmp = safe_read_csv(f_path)
            if df_tmp is not None and len(df_tmp.columns) >= len(new_cols):
                df_tmp = df_tmp.iloc[:, :len(new_cols)]
                df_tmp.columns = new_cols
                df_tmp['date'] = df_tmp['date'].apply(parse_japanese_date)
                all_data.append(df_tmp)
        if all_data:
            df_all_g = pd.concat(all_data).dropna(subset=['date']).sort_values('date')
            for c in calc_cols: df_all_g[c] = pd.to_numeric(df_all_g[c], errors='coerce')
            st.subheader("生長推移データ")
            
            latest_date = df_all_g['date'].max()
            df_latest = df_all_g[df_all_g['date'] == latest_date]
            st.markdown(f"### 【最新生育調査データ: {latest_date.strftime('%Y/%m/%d')}】")
            c1, c2 = st.columns([1, 1.5])
            with c1: st.write("**全体平均**"); st.table(df_latest[calc_cols].mean().to_frame().T.style.format("{:.1f}"))
            with c2: st.write("**エリア別平均**"); st.dataframe(df_latest.groupby('area')[calc_cols].mean().style.format("{:.1f}"), use_container_width=True)
            
            st.divider(); st.write("#### 【生育推移】")
            df_trend = df_all_g.groupby('date')[calc_cols].mean().reset_index()
            # 日付を文字列に変換してCategorical Axisとして扱う
            df_trend['date_str'] = df_trend['date'].dt.strftime('%m/%d')
            
            # レイアウト: 左(表) / 右(グラフ)
            # 比率を 1:1 に変更してグラフ幅を縮小
            col_t_left, col_t_right = st.columns([1, 1])
            
            with col_t_left:
                st.write("##### 推移データ表（15株平均）")
                # 日付と主要項目を表示
                disp_cols_trend = ['date'] + calc_cols
                df_trend_disp = df_trend.copy()
                df_trend_disp['date'] = df_trend_disp['date'].dt.strftime('%Y/%m/%d')
                # フォーマット設定 (数値列のみ)
                format_dict = {c: "{:.1f}" for c in calc_cols}
                st.dataframe(df_trend_disp.rename(columns={'date':'日付'}).set_index('日付').style.format(format_dict), use_container_width=True, height=500)

            with col_t_right:
                st.write("##### 推移グラフ")
                # 表示項目の選択
                metrics_map = {
                    '草高(cm)': 'royalblue', '第3葉柄長(cm)': 'lightseagreen', 
                    '葉数': 'orange', '蕾数': 'pink', '花数': 'red', 
                    '肥大果数': 'green', '緑熟果数': 'lightgreen', '白熟果数': 'white'
                }
                # デフォルト選択
                default_metrics = ['草高(cm)', '第3葉柄長(cm)', '葉数']
                
                selected_growth_metrics = st.multiselect("グラフに表示する項目を選択", options=list(metrics_map.keys()), default=default_metrics)
                
                if selected_growth_metrics:
                    fig_growth = go.Figure()
                    for m in selected_growth_metrics:
                        fig_growth.add_trace(go.Scatter(
                            x=df_trend['date_str'], y=df_trend[m], name=m, 
                            line=dict(color=metrics_map.get(m, 'gray'), width=3, shape='spline')
                        ))
                    
                    fig_growth.update_layout(
                        height=500, # 表の高さに合わせる
                        margin=dict(t=20, b=50), 
                        template="plotly_white", 
                        xaxis=dict(title="日付", type='category', tickangle=-45),
                        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)
                else:
                    st.warning("項目を選択してください")
    else: st.info("データ待機中: data/growth/ フォルダにCSVを配置してください。")

# --- TAB 2: 環境トレンド (天気連携版) ---
with tab2:
    if df_u is not None:
        st.subheader("環境推移データ")
        dt_col = df_u.columns[0]
        df_u[dt_col] = pd.to_datetime(df_u[dt_col], errors='coerce')
        df_u = df_u.dropna(subset=[dt_col]).sort_values(dt_col)
        df_u['飽差'] = df_u.apply(lambda x: calculate_vpd(x['室内気温[C]'], x['室内相対湿度[%]']), axis=1)
        
        min_date = df_u[dt_col].min()
        max_date = df_u[dt_col].max()

        # 天気データ取得 (Open-Meteo) w/ キャッシュ的な挙動のためにsession_stateを使いたいが、
        # 簡易的に毎回取得（期間が変わる可能性があるため）
        with st.spinner("外気温データを取得中(Open-Meteo)..."):
            df_weather_history = fetch_weather_data(lat, lon, start_date=min_date, end_date=max_date)

        all_metrics = {
            '気温(℃)': {'col': '室内気温[C]', 'color': '#FF0000', 'range': [0, 45], 'unit': '℃'},
            '湿度(%)': {'col': '室内相対湿度[%]', 'color': '#00008B', 'range': [0, 100], 'unit': '%'},
            'CO2(ppm)': {'col': '室内CO2濃度[ppm]', 'color': '#32CD32', 'range': [0, 2500], 'unit': 'ppm'},
            '日射(kW)': {'col': '室内日射強度[kW m-2]', 'color': '#FFA500', 'range': [0, 1.5], 'unit': 'kW'},
            '積算日射(MJ)': {'col': '積算日射[MJ]', 'color': '#00BFFF', 'range': [0, 3.0], 'unit': 'MJ'},
            '飽差(g/m³)': {'col': '飽差', 'color': '#9400D3', 'range': [0, 15], 'unit': 'g/m³'}
        }
        
        # 選択肢に「外気温（API連携）」を追加
        overlay_weather = st.checkbox("外気温（Open-Meteo）を重ねて表示", value=True)
        
        selected_metrics = st.multiselect(
            "表示する項目を選択", options=list(all_metrics.keys()), default=list(all_metrics.keys())
        )
        
        fig = go.Figure()
        y_axis_map = {}
        
        # ハウス内データの描画
        for i, metric_name in enumerate(selected_metrics):
            y_name = f"y{i+1}" if i > 0 else "y"
            y_axis_map[metric_name] = f"yaxis{i+1}" if i > 0 else "yaxis"
            meta = all_metrics[metric_name]
            if meta['col'] in df_u.columns:
                fig.add_trace(go.Scatter(
                    x=df_u[dt_col], y=df_u[meta['col']], name=metric_name,
                    line=dict(color=meta['color'], width=2, shape='spline'),
                    yaxis=y_name
                ))
                
        # 外気温の描画 (重ね合わせ: 気温の軸を使用)
        if overlay_weather and df_weather_history is not None and '気温(℃)' in selected_metrics:
            # 気温軸を探す
            temp_y_axis = "y" # デフォルト
            for m, ax in zip(selected_metrics, [f"y{i+1}" if i>0 else "y" for i in range(len(selected_metrics))]):
                if m == '気温(℃)':
                    temp_y_axis = ax
                    break
            
            fig.add_trace(go.Scatter(
                x=df_weather_history['time'], y=df_weather_history['temp_out'], 
                name="外気温(API)",
                line=dict(color='gray', width=2, dash='dot'), # 点線で表示
                yaxis=temp_y_axis,
                hoverinfo='y+name'
            ))

        # 背景(夜間)
        check_date = min_date.date()
        while check_date <= max_date.date():
            dt_base = datetime(check_date.year, check_date.month, check_date.day)
            sr, ss = get_sun_times(dt_base, lat, lon)
            fig.add_vrect(x0=dt_base, x1=dt_base + timedelta(hours=sr), fillcolor="gray", opacity=0.1, layer="below", line_width=0)
            fig.add_vrect(x0=dt_base + timedelta(hours=ss), x1=dt_base + timedelta(days=1), fillcolor="gray", opacity=0.1, layer="below", line_width=0)
            check_date += timedelta(days=1)

        # レイアウト
        num_vars = len(selected_metrics)
        left_margin_size = max(0.06 * num_vars, 0.05) 
        domain_start = left_margin_size + 0.02
        
        layout_updates = {
            'height': 600, 'hovermode': "x unified", 'template': "plotly_white",
            'xaxis': dict(
                domain=[domain_start, 1.0], 
                rangeselector=dict(
                    buttons=list([
                        dict(count=24, label="24H", step="hour", stepmode="backward"),
                        dict(count=3, label="3Days", step="day", stepmode="backward"),
                        dict(count=7, label="1Week", step="day", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]), x=0, y=1.2
                ),
                rangeslider=dict(visible=False), type="date"
            ),
            'legend': dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            'margin': dict(l=20, r=20, t=80, b=80)
        }
        
        for i, metric_name in enumerate(selected_metrics):
            y_key = f"yaxis{i+1}" if i > 0 else "yaxis"
            meta = all_metrics[metric_name]
            pos = domain_start - (0.06 * (i + 1))
            axis_settings = dict(
                title=dict(text=meta['unit'], font=dict(color=meta['color'])),
                tickfont=dict(color=meta['color']),
                range=meta['range'],
                side="left", position=pos, anchor="free", showgrid=(i==0),
            )
            if i > 0: axis_settings['overlaying'] = 'y'
            layout_updates[y_key] = axis_settings

        fig.update_layout(**layout_updates)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: 天気予報 (New) ---
with tab3:
    st.subheader("週間天気予報 (by Open-Meteo)")
    
    with st.spinner("天気予報を取得中..."):
        df_forecast = fetch_weather_data(lat, lon) # 期間指定なし＝7日間予報
    
    if df_forecast is not None:
        # 見やすく加工
        df_forecast['日付'] = df_forecast['time'].dt.strftime('%m/%d %H:00')
        df_forecast['天気'] = df_forecast['weather_code'].apply(get_weather_icon)
        
        # グラフ: 気温と降水確率(今回は湿度を表示)の予測
        fig_f = make_subplots(specs=[[{"secondary_y": True}]])
        fig_f.add_trace(go.Scatter(x=df_forecast['time'], y=df_forecast['temp_out'], name="予想気温(℃)", line=dict(color='red')), secondary_y=False)
        fig_f.add_trace(go.Bar(x=df_forecast['time'], y=df_forecast['humid_out'], name="予想湿度(%)", marker_color='blue', opacity=0.3), secondary_y=True)
        
        fig_f.update_layout(title="週間予報 (気温 & 湿度)", height=400, template="plotly_white")
        fig_f.update_yaxes(title_text="気温(℃)", secondary_y=False)
        fig_f.update_yaxes(title_text="湿度(%)", range=[0, 100], secondary_y=True)
        st.plotly_chart(fig_f, use_container_width=True)
        
        # 詳細テーブル
        st.write("#### 詳細データ")
        disp_cols = ['日付', '天気', 'temp_out', 'humid_out', 'wind_speed']
        df_disp = df_forecast[disp_cols].rename(columns={
            'temp_out': '気温(℃)', 'humid_out': '湿度(%)', 'wind_speed': '風速(m/s)'
        })
        st.dataframe(df_disp, use_container_width=True)
    else:
        st.error("天気予報データを取得できませんでした。")

# --- TAB 4: AI Agronomist (New) ---
with tab4:
    st.markdown("### 🤖 AI Agronomist (Beta)")
    st.write("Gemini AIが現在の環境データと生育データを分析し、栽培アドバイスを行います。")
    
    if not gemini_api_key:
        st.warning("⚠️ サイドバーで Gemini API Key を設定してください。")
    else:
        # 1. 自動分析ボタン
        if st.button("📊 現在の状況を分析する"):
            with st.spinner("データを集計してAIに送信中..."):
                # コンテキスト作成
                context_summary = "【環境データ概要】\n"
                if df_u is not None:
                    latest_env = df_u.iloc[-1]
                    context_summary += f"- 最新日時: {latest_env[dt_col]}\n"
                    context_summary += f"- 現在気温: {latest_env.get('室内気温[C]', 'N/A')} ℃\n"
                    context_summary += f"- 現在湿度: {latest_env.get('室内相対湿度[%]', 'N/A')} %\n"
                    context_summary += f"- 現在CO2: {latest_env.get('室内CO2濃度[ppm]', 'N/A')} ppm\n"
                    context_summary += f"- 現在日射: {latest_env.get('室内日射強度[kW m-2]', 'N/A')} kW\n"
                    context_summary += f"- 本日積算日射: {latest_env.get('積算日射[MJ]', 'N/A')} MJ\n"
                
                context_summary += "\n【生育データ概要】\n"
                if growth_files and 'df_latest' in locals():
                     avg_growth = df_latest[calc_cols].mean()
                     context_summary += f"- 草高: {avg_growth.get('草高(cm)', 'N/A'):.1f} cm\n"
                     context_summary += f"- 葉数: {avg_growth.get('葉数', 'N/A'):.1f} 枚\n"
                     context_summary += f"- 開花数: {avg_growth.get('花数', 'N/A'):.1f} \n"
                
                system_prompt = f"""
                あなたはプロの施設園芸アドバイザー（Agronomist）です。以下のイチゴ（Strawberry）栽培データに基づいて、現在の状況評価と管理アドバイスを行ってください。
                
                {context_summary}
                
                回答は以下の形式で簡潔にお願いします：
                1. 現状の評価（良い点・気になる点）
                2. 今後の管理アドバイス（環境制御・作業など）
                """
                
                response_text = get_ai_response(gemini_api_key, system_prompt, "分析をお願いします")
                st.session_state.chat_history.append({"role": "user", "content": "現状分析をお願いします"})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # 2. チャット履歴の表示
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. チャット入力
        if prompt := st.chat_input("栽培について質問する..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("AIが生活中..."):
                # コンテキストを維持するために簡易的に前の情報を付与（本来はChatSessionを使う）
                context_summary_short = "（※直前のデータ分析結果を踏まえて回答してください）"
                response = get_ai_response(gemini_api_key, context_summary_short, prompt)
                
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
