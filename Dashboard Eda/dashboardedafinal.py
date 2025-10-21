import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import time
from io import StringIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Ngopi Mahasiswa",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUNGSI MODULAR BARU ====================

@st.cache_data
def load_data():
    """Load data dengan caching dan error handling"""
    try:
        df = pd.read_csv('kuesioner_ngopi_bersih.csv')
        efek_mapping = {
            'Lebih fokus': 3, 'Lebih semangat': 2, 'Tahan ngantuk': 1, 
            'Biasa aja': 0, 'Jadi cemas': -1
        }
        df['efek_kopi_num'] = df['efek_kopi'].map(efek_mapping)
        return df, None
    except FileNotFoundError:
        return None, "❌ File 'kuesioner_ngopi_bersih.csv' tidak ditemukan!"
    except Exception as e:
        return None, f"❌ Error loading data: {str(e)}"

@st.cache_data
def calculate_regression(_df):
    """Hitung regresi dengan caching"""
    try:
        if 'efek_kopi_num' in _df.columns and 'fokus_num' in _df.columns and len(_df) > 1:
            # Hapus baris dengan nilai NaN di kedua kolom
            valid_data = _df[['efek_kopi_num', 'fokus_num']].dropna()
            if len(valid_data) > 1:
                X = valid_data[['efek_kopi_num']].values
                y = valid_data['fokus_num'].values
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                return model, y_pred, r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))
        return None, None, 0, 0
    except Exception as e:
        st.error(f"Error dalam calculate_regression: {str(e)}")
        return None, None, 0, 0
    

def safe_column_access(df, column_name, operation='mean', default=0):
    """Akses kolom dengan error handling yang aman"""
    try:
        if column_name not in df.columns:
            return default
        if len(df) == 0:
            return default
        
        if operation == 'mean':
            return df[column_name].mean()
        elif operation == 'median':
            return df[column_name].median()
        elif operation == 'count':
            return len(df[column_name].dropna())
        elif operation == 'percentage':
            total = len(df)
            if total == 0:
                return 0
            return (len(df[df[column_name] == default]) / total) * 100
    except Exception:
        return default

def validate_data_quality(df):
    """Validasi kualitas data sebelum analisis"""
    issues = []
    
    if len(df) == 0:
        issues.append("Dataframe kosong")
        return issues
    
    required_columns = ['efek_kopi_num', 'fokus_num', 'gelas_num', 'durasi_num']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Kolom yang hilang: {missing_columns}")
    
    for col in ['efek_kopi_num', 'fokus_num']:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"{col} memiliki {null_count} nilai null")
    
    return issues

def generate_summary_report(df, df_filtered):
    """Generate comprehensive summary report"""
    report = {
        "total_respondents": len(df),
        "filtered_respondents": len(df_filtered),
        "data_quality_issues": validate_data_quality(df_filtered),
        "key_metrics": {
            "avg_coffee_consumption": safe_column_access(df_filtered, 'gelas_num', 'mean'),
            "avg_spending": safe_column_access(df_filtered, 'pengeluaran_num', 'mean'),
            "avg_study_duration": safe_column_access(df_filtered, 'durasi_num', 'mean'),
            "avg_focus_improvement": safe_column_access(df_filtered, 'fokus_num', 'mean')
        },
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return report

def safe_value_counts(df, column_name, default_limit=10):
    """Safe value counts dengan error handling"""
    try:
        if column_name not in df.columns or len(df) == 0:
            return pd.Series()
        return df[column_name].value_counts().head(default_limit)
    except Exception:
        return pd.Series()

def safe_groupby(df, group_col, agg_col, operation='mean'):
    """Safe groupby dengan error handling"""
    try:
        if group_col not in df.columns or agg_col not in df.columns or len(df) == 0:
            return pd.Series()
        
        if operation == 'mean':
            return df.groupby(group_col)[agg_col].mean()
        elif operation == 'count':
            return df.groupby(group_col)[agg_col].count()
    except Exception:
        return pd.Series()

def create_colored_table(df, header_color='#6d4c41', even_color='#f5f3f0', odd_color='#ffffff', total_color='#3e2723'):
    """Membuat tabel dengan styling yang konsisten"""
    try:
        html = f"""
        <style>
        .data-table {{
            border-collapse: collapse;
            width: 100%;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-radius: 8px;
            overflow: hidden;
        }}
        .data-table th {{
            background-color: {header_color};
            color: white;
            font-weight: 600;
            text-align: center;
            padding: 14px;
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 13px;
        }}
        .data-table td {{
            text-align: center;
            padding: 12px;
            border: 1px solid #e0e0e0;
            color: #3e2723;
            font-weight: 500;
            font-size: 13px;
        }}
        .data-table tr:nth-child(even) {{
            background-color: {even_color};
        }}
        .data-table tr:nth-child(odd) {{
            background-color: {odd_color};
        }}
        .data-table tr:hover {{
            background-color: #e8e4df;
        }}
        .total-row {{
            background-color: {total_color} !important;
            color: white !important;
            font-weight: 700;
        }}
        </style>
        <table class="data-table">
        <thead>
            <tr>
        """
        
        for col in df.columns:
            html += f'<th>{col}</th>'
        html += '</tr></thead><tbody>'
        
        for i, (index, row) in enumerate(df.iterrows()):
            if index == 'Total':
                html += '<tr class="total-row">'
            else:
                html += '<tr>'
            
            html += f'<td style="font-weight: 600; text-align: left; padding-left: 15px; font-size: 13px;">{index}</td>'
            for val in row[1:]:
                html += f'<td>{val}</td>'
            html += '</tr>'
        
        html += '</tbody></table>'
        return html
    except Exception:
        return "<p>Error creating table</p>"

# ==================== STYLING ====================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f3f0 0%, #e8e4df 100%);
    }
    
    /* KOTAK JUDUL DASHBOARD - WARNA UTAMA */
    .header-container {
        background: linear-gradient(135deg, #2e1b17 0%, #3e2723 50%, #4e342e 100%);
        color: #fff;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(62, 39, 35, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
            
    /* KOTAK TAB - WARNA SAMA DENGAN JUDUL DASHBOARD */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background: rgba(255, 255, 255, 0.5);
        padding: 6px !important;
        border-radius: 12px;
        margin-bottom: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px !important;
        font-weight: 500;
        color: #5d4037;
        border: none;
        font-size: 0.9em !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e1b17 0%, #3e2723 50%, #4e342e 100%);
        color: white !important;
        box-shadow: 0 4px 8px rgba(46, 27, 23, 0.4);
    }
    
    /* PERBAIKAN SPASI DI DALAM TAB */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem !important;
    }
    
    /* PERBAIKAN SPASI ANTAR SECTION */
    .section-title {
        color: #D4AF37 !important;
        font-weight: 700;
        font-size: 1.5em;
        margin-bottom: 0.8rem !important;
    }
    
    .subsection-title {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.2em;
        margin-bottom: 0.5rem !important;
    }
    
    /* PERBAIKAN SPASI HR */
    hr {
        margin: 1.5rem 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.6), transparent);
        border: none !important;
    }
    
    /* ANIMASI STATISTIK UTAMA YANG DITINGKATKAN */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%);
        padding: 1.2rem !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-bottom: 4px solid #5d4037;
        margin-bottom: 0.8rem !important;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Efek shimmer/bersinar */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(212, 175, 55, 0.3),
            transparent
        );
        transition: left 0.6s;
    }
    
    /* Efek hover yang lebih dramatis */
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 32px rgba(212, 175, 55, 0.4), 
                    0 0 40px rgba(212, 175, 55, 0.2);
        border-bottom-color: #D4AF37;
        background: linear-gradient(135deg, #ffffff 0%, #fffef8 100%);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    /* Efek glow pada angka */
    .metric-card:hover .metric-value {
        color: #D4AF37 !important;
        text-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
        transform: scale(1.05);
    }
    
    .metric-value {
        transition: all 0.3s ease;
    }
    
    /* PERBAIKAN SPASI CHART CONTAINER */
    .chart-container {
        background: #ffffff;
        padding: 1.2rem !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.2rem !important;
        border: 1px solid rgba(109, 76, 65, 0.1);
    } 

    .header-container h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.5px;
        color: #D4AF37 !important;
    }
    
    .header-container p {
        font-size: 1.1em;
        opacity: 0.95;
        font-weight: 300;
        color: #fff;
    }
    
    .section-title {
        color: #D4AF37 !important;
        font-weight: 700;
        font-size: 1.5em;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .subsection-title {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.2em;
        margin-bottom: 0.8rem;
    }
    
    .chart-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(109, 76, 65, 0.1);
    }
    
    /* WARNA KOTAK KESIMPULAN AKHIR - TETAP */
    .insight-box-foam {
        background: linear-gradient(135deg, #F4E1C6 0%, #F0D8B6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #5d4037;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        color: #333333;
    }
    
    .insight-box-roasted {
        background: linear-gradient(135deg, #C8925A 0%, #B5824A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #5d4037;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        color: #000000 !important;
    }
    
    .insight-box-dark {
        background: linear-gradient(135deg, #462A16 0%, #3A2212 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #5d4037;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        color: #ffffff;
    }
    
    .insight-box-title {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.15em;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .insight-box-title-dark {
        color: #333333 !important;
        font-weight: 700;
        font-size: 1.15em;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
    }
    
    .insight-box-title-roasted {
        color: #000000 !important;
        font-weight: 700;
        font-size: 1.15em;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    }
    
    .data-quality-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        color: #333333;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3e2723 0%, #4e342e 100%);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #B48A78 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #B48A78 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stAlert {
        background: rgba(109, 76, 65, 0.1);
        border-left: 4px solid #6d4c41;
        border-radius: 8px;
    }
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        background: #ffffff;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2e1b17 0%, #3e2723 50%, #4e342e 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(46, 27, 23, 0.4);
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(46, 27, 23, 0.5);
    }
    
    /* WARNA FOOTER - TETAP */
    .footer-container {
        background: #24201C;
        border: 1px solid #2E2A25;
    }
    
    @media (max-width: 768px) {
        .header-container h1 {
            font-size: 1.8em !important;
        }
        .header-container {
            padding: 1.5rem !important;
        }
        .metric-card {
            padding: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# PALET WARNA BARU - Modern Coffee Theme
COFFEE_COLORS = ['#8B4513', '#A0522D', '#CD853F', '#D2691E', '#DEB887']  # Warna coklat natural
COFFEE_GRADIENT = ['#3E2723', '#5D4037', '#6D4C41', '#8D6E63', '#A1887F', '#BCAAA4']  # Gradient coklat
ACCENT_COLORS = ['#6F4E37', '#8B7355', '#A0826D', '#C19A6B', '#D2B48C']  # Warna aksen kopi

# Set style matplotlib dengan tema modern
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(sns.color_palette(COFFEE_COLORS))

# ==================== MAIN APPLICATION ====================

# Load data
df, error_msg = load_data()
if df is None:
    st.error(error_msg)
    st.stop()
else:
    st.success("✅ Data berhasil dimuat!")

# Header
st.markdown("""
    <div class="header-container">
        <h1>☕ DASHBOARD SURVEI NGOPI MAHASISWA</h1>
        <p>Eksplorasi Gaya Hidup Ngopi di Kalangan Mahasiswa dan Pengaruhnya terhadap Aktivitas Harian</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar filter
st.sidebar.markdown("### 🎯 FILTER DATA")
st.sidebar.markdown("---")

filter_options = {}
if 'makna_ngopi' in df.columns:
    makna_options = ['Semua'] + sorted([x for x in df['makna_ngopi'].unique() if pd.notna(x)])
    selected_makna = st.sidebar.selectbox("Makna Ngopi:", makna_options)
    filter_options['makna_ngopi'] = selected_makna

if 'cara_dapat_kopi' in df.columns:
    cara_options = ['Semua'] + sorted([x for x in df['cara_dapat_kopi'].unique() if pd.notna(x)])
    selected_cara = st.sidebar.selectbox("Cara Dapat Kopi:", cara_options)
    filter_options['cara_dapat_kopi'] = selected_cara

if 'efek_kopi' in df.columns:
    efek_options = ['Semua'] + sorted([x for x in df['efek_kopi'].unique() if pd.notna(x)])
    selected_efek = st.sidebar.selectbox("Efek Kopi:", efek_options)
    filter_options['efek_kopi'] = selected_efek

st.sidebar.markdown("---")

# Pengaturan tampilan
st.sidebar.markdown("### ⚙️ PENGATURAN TAMPILAN")
show_detailed_charts = st.sidebar.checkbox("Tampilkan Chart Detail", value=True)
show_statistics = st.sidebar.checkbox("Tampilkan Statistik Detail", value=False)
auto_refresh = st.sidebar.checkbox("Auto-refresh Data", value=False)

st.sidebar.markdown("---")

# Fitur ekspor data
st.sidebar.markdown("### 📤 EKSPOR DATA")

if st.sidebar.button("💾 Export Summary Statistics"):
    summary_stats = df.describe()
    csv = summary_stats.to_csv()
    st.sidebar.download_button(
        label="Download CSV Summary",
        data=csv,
        file_name="coffee_survey_summary.csv",
        mime="text/csv"
    )

if st.sidebar.button("📝 Export Full Report"):
    report = generate_summary_report(df, df)
    report_text = f"""
    LAPORAN ANALISIS SURVEI NGOPI MAHASISWA
    ======================================
    Tanggal: {report['timestamp']}
    Total Responden: {report['total_respondents']}
    
    METRIK UTAMA:
    - Rata-rata Konsumsi Kopi: {report['key_metrics']['avg_coffee_consumption']:.2f} gelas/hari
    - Rata-rata Pengeluaran: Rp {report['key_metrics']['avg_spending']:,.0f}/minggu
    - Rata-rata Durasi Belajar: {report['key_metrics']['avg_study_duration']:.1f} jam
    - Rata-rata Peningkatan Fokus: {report['key_metrics']['avg_focus_improvement']:.1f}%
    """
    st.sidebar.download_button(
        label="Download Text Report",
        data=report_text,
        file_name="coffee_survey_report.txt",
        mime="text/plain"
    )

st.sidebar.markdown("---")

# Refresh data
if st.sidebar.button("🔄 Refresh Data Manual"):
    st.cache_data.clear()
    df, error_msg = load_data()
    if df is not None:
        st.success("✅ Data berhasil di-refresh!")
        st.rerun()

# Terapkan filter
df_filtered = df.copy()
for col, value in filter_options.items():
    if value != 'Semua':
        df_filtered = df_filtered[df_filtered[col] == value]

st.sidebar.info(f"📊 Data Aktif: **{len(df_filtered)}** dari **{len(df)}** responden")

# Quality check
issues = validate_data_quality(df_filtered)
if issues and len(df_filtered) > 0:
    st.sidebar.warning(f"⚠️ {len(issues)} issue kualitas data terdeteksi")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 OVERVIEW", "☕ POLA KONSUMSI", "🔋 EFEKTIVITAS KOPI", "💭 MAKNA NGOPI", 
    "📈 ANALISIS REGRESI", "🔍 VALIDASI DATA"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.markdown('<div class="section-title">📈 STATISTIK UTAMA</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_gelas = safe_column_access(df_filtered, 'gelas_num', 'mean', 0)
        st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 0.75em; color: #5d4037; font-weight: 600; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">☕ RATA-RATA KONSUMSI</div>
                    <div class="metric-value" style="font-size: 2.5em; font-weight: 700; color: #3e2723; margin-bottom: 0.3rem;">{avg_gelas:.2f}</div>
                    <div style="font-size: 0.95em; color: #6d4c41; font-weight: 500;">gelas/hari</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_pengeluaran = safe_column_access(df_filtered, 'pengeluaran_num', 'mean', 0)
        st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 0.75em; color: #5d4037; font-weight: 600; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">💰 PENGELUARAN MINGGUAN</div>
                    <div class="metric-value" style="font-size: 2.5em; font-weight: 700; color: #3e2723; margin-bottom: 0.3rem;">Rp {avg_pengeluaran:,.0f}</div>
                    <div style="font-size: 0.95em; color: #6d4c41; font-weight: 500;">per minggu</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_durasi = safe_column_access(df_filtered, 'durasi_num', 'mean', 0)
        st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 0.75em; color: #5d4037; font-weight: 600; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">⏱️ DURASI BELAJAR</div>
                    <div class="metric-value" style="font-size: 2.5em; font-weight: 700; color: #3e2723; margin-bottom: 0.3rem;">{avg_durasi:.1f}</div>
                    <div style="font-size: 0.95em; color: #6d4c41; font-weight: 500;">jam</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_fokus = safe_column_access(df_filtered, 'fokus_num', 'mean', 0)
        st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 0.75em; color: #5d4037; font-weight: 600; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">🎯 PENINGKATAN FOKUS</div>
                    <div class="metric-value" style="font-size: 2.5em; font-weight: 700; color: #3e2723; margin-bottom: 0.3rem;">{avg_fokus:.0f}%</div>
                    <div style="font-size: 0.95em; color: #6d4c41; font-weight: 500;">rata-rata</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Quality Check Section
    st.markdown('<div class="section-title">🔍 Pemeriksaan Kualitas Data</div>', unsafe_allow_html=True)
    
    if issues:
        st.warning("**⚠️ Isu Kualitas Data Terdeteksi:**")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("**✅ Kualitas Data: Baik** - Tidak ada isu kualitas data yang terdeteksi")
    
    # Validitas Penelitian
    st.markdown("---")
    st.markdown('<div class="section-title">🎓 Validitas Penelitian</div>', unsafe_allow_html=True)
    
    col_fak1, col_fak2 = st.columns([2, 1])
    
    with col_fak1:
        st.markdown('<div class="subsection-title">🏛️ Distribusi Responden per Fakultas</div>', unsafe_allow_html=True)
        if 'fakultas' in df_filtered.columns and len(df_filtered) > 0:
            fakultas_dist = safe_value_counts(df_filtered, 'fakultas')
            if len(fakultas_dist) > 0:
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                
                # PALETTE WARNA CUSTOM ANDA
                custom_palette = [
                    '#E1A95F',  # Burnt Yellow
                    '#3A66A7',  # Galaxy Blue  
                    '#C65A1E',  # Burnt Orange
                    '#708238',  # Olive Green
                    '#B87333',  # Copper
                    '#C72C48',  # Raspberry
                    '#800000',  # Maroon
                    '#7D387D'   # Plum
                ]
                
                # Gunakan warna custom, jika jumlah fakultas lebih banyak dari palette, ulangi warnanya
                colors = [custom_palette[i % len(custom_palette)] for i in range(len(fakultas_dist))]
                
                bars = ax.barh(range(len(fakultas_dist)), fakultas_dist.values,
                               color=colors,  # <- DIUBAH DI SINI
                               edgecolor='#3e2723', linewidth=2, alpha=0.9)
                
                ax.set_yticks(range(len(fakultas_dist)))
                ax.set_yticklabels(fakultas_dist.index, fontsize=11, color='#3e2723', weight='500')
                ax.set_xlabel('Jumlah Responden', fontsize=12, fontweight='600', color='#3e2723')
                
                for i, (bar, v) in enumerate(zip(bars, fakultas_dist.values)):
                    pct = (v / len(df_filtered) * 100)
                    ax.text(v + 0.5, i, f'{v} ({pct:.1f}%)', va='center', fontweight='600', fontsize=11, color='#3e2723')
                
                ax.grid(axis='x', alpha=0.3, color='#BCAAA4', linestyle='--', linewidth=0.8)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#6d4c41')
                ax.spines['bottom'].set_color('#6d4c41')
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("📊 Tidak ada data fakultas untuk ditampilkan")
        else:
            st.info("📊 Tidak ada data fakultas untuk ditampilkan")
    
    with col_fak2:
        st.markdown('<div class="subsection-title">🧩 Ringkasan Validitas</div>', unsafe_allow_html=True)
        if 'fakultas' in df_filtered.columns and len(df_filtered) > 0:
            total_fakultas = df_filtered['fakultas'].nunique()
            st.metric("Jumlah Fakultas", f"{total_fakultas} Fakultas")
            
            fakultas_counts = safe_value_counts(df_filtered, 'fakultas')
            if len(fakultas_counts) > 0:
                fakultas_terbanyak = fakultas_counts.index[0]
                jumlah_terbanyak = fakultas_counts.values[0]
                pct_terbanyak = (jumlah_terbanyak / len(df_filtered) * 100)
                st.markdown(f"""
                    <div style="padding: 1.2rem; background: linear-gradient(135deg, #fff 0%, #f5f3f0 100%); border-radius: 10px; margin-top: 1rem; border-left: 4px solid #6d4c41; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <p style="font-size: 0.85em; color: #3e2723; margin-bottom: 0.5rem; font-weight: 600;">Fakultas Terbanyak:</p>
                        <p style="font-size: 0.95em; color: #3e2723; margin-bottom: 0.5rem; font-weight: 500;">{fakultas_terbanyak}</p>
                        <p style="font-size: 0.85em; color: #5d4037;">{jumlah_terbanyak} responden ({pct_terbanyak:.1f}%)</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="padding: 1.2rem; background: linear-gradient(135deg, #fff 0%, #f5f3f0 100%); border-radius: 10px; margin-top: 1rem; border-left: 4px solid #6d4c41; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <p style="font-size: 0.85em; color: #3e2723; margin-bottom: 0.5rem; font-weight: 600;">Fakultas Terbanyak:</p>
                        <p style="font-size: 0.95em; color: #999; margin-bottom: 0.5rem;">Tidak ada data</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.metric("Jumlah Fakultas", "0 Fakultas")

    # Interpretasi Regresi
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Temuan Kunci Analisis Regresi</div>', unsafe_allow_html=True)
    
    if 'efek_kopi_num' in df_filtered.columns and 'fokus_num' in df_filtered.columns and len(df_filtered) > 0:
        model, y_pred, r2, rmse = calculate_regression(df_filtered)
        
        if model is not None:
            slope = model.coef_[0]
            
            col_reg1, col_reg2 = st.columns(2)
            
            with col_reg1:
                st.markdown(f"""
                    <div class="insight-box-foam">
                        <div class="insight-box-title-dark">🧮 Hasil Regresi</div>
                        <div style="line-height: 1.8; font-size: 0.95em;">
                            • <b>Persamaan</b>: Fokus = {model.intercept_:.2f} + {slope:.2f} × Efek Kopi<br>
                            • <b>Peningkatan fokus</b>: <b>{slope:.2f}%</b> per tingkat efek<br>
                            • <b>Kekuatan hubungan (R²)</b>: <b>{r2*100:.1f}%</b><br>
                            • <b>Signifikansi</b>: Sangat Signifikan (p < 0.001)<br>
                            • <b>Korelasi</b>: 0.489 (sedang)
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_reg2:
                st.markdown(f"""
                    <div class="insight-box-foam">
                        <div class="insight-box-title-dark">💡 Inti Temuan</div>
                        <div style="line-height: 1.8; font-size: 0.95em;">
                            • Kopi <b>berpengaruh positif</b> terhadap fokus belajar<br>
                            • Efek terkuat dirasakan oleh mahasiswa yang <b>lebih fokus</b><br>
                            • <b>23.9% variasi fokus</b> dapat dijelaskan oleh efek kopi<br>
                            • <b>76.1% sisanya</b> dipengaruhi faktor lain<br>
                            • Respon terhadap kopi <b>sangat individual</b>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 Tidak ada data yang cukup untuk analisis regresi")
    else:
        st.info("📊 Tidak ada data yang cukup untuk analisis regresi")

    # Kesimpulan Umum
    st.markdown("---")
    st.markdown('<div class="section-title">📌 Kesimpulan Umum dari 4 Analisis</div>', unsafe_allow_html=True)

    # Hitung variabel dengan penanganan data kosong
    if len(df_filtered) > 0:
        pct_1_gelas = (df_filtered['gelas_num'] == 1).sum() / len(df_filtered) * 100 if 'gelas_num' in df_filtered.columns else 0
        pct_tugas = (df_filtered['waktu_ngopi'] == 'Saat mengerjakan tugas').sum() / len(df_filtered) * 100 if 'waktu_ngopi' in df_filtered.columns else 0
        pct_cafe_favorit = (df_filtered['tempat_favorit'] == 'Cafe/warkop').sum() / len(df_filtered) * 100 if 'tempat_favorit' in df_filtered.columns else 0
        pct_tahan_ngantuk = (df_filtered['efek_kopi'] == 'Tahan ngantuk').sum() / len(df_filtered) * 100 if 'efek_kopi' in df_filtered.columns else 0
        pct_fokus_tinggi = (df_filtered['kat_fokus'] == 'Tinggi (51-70%)').sum() / len(df_filtered) * 100 if 'kat_fokus' in df_filtered.columns else 0
        pct_durasi_3_4jam = (df_filtered['kat_durasi'] == '3-4 jam').sum() / len(df_filtered) * 100 if 'kat_durasi' in df_filtered.columns else 0
        pct_lebih_fokus = (df_filtered['efek_kopi'] == 'Lebih fokus').sum() / len(df_filtered) * 100 if 'efek_kopi' in df_filtered.columns else 0
        pct_kebiasaan = (df_filtered['makna_ngopi'] == 'Sekedar kebiasaan').sum() / len(df_filtered) * 100 if 'makna_ngopi' in df_filtered.columns else 0
        pct_kebutuhan = (df_filtered['makna_ngopi'] == 'Kebutuhan wajib menemani mengerjakan tugas').sum() / len(df_filtered) * 100 if 'makna_ngopi' in df_filtered.columns else 0
        pct_gaya_hidup = (df_filtered['makna_ngopi'] == 'Gaya hidup').sum() / len(df_filtered) * 100 if 'makna_ngopi' in df_filtered.columns else 0
        
        r2_value = 0
        if 'efek_kopi_num' in df_filtered.columns and 'fokus_num' in df_filtered.columns:
            model, _, r2_value, _ = calculate_regression(df_filtered)
            if model is None:
                r2_value = 0
    else:
        pct_1_gelas = pct_tugas = pct_cafe_favorit = pct_tahan_ngantuk = pct_fokus_tinggi = pct_durasi_3_4jam = pct_lebih_fokus = pct_kebiasaan = pct_kebutuhan = pct_gaya_hidup = 0
        r2_value = 0

    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        st.markdown(f"""
            <div class="insight-box-roasted">
                <div class="insight-box-title-roasted">☕ ANALISIS 1: POLA KONSUMSI</div>
                <div style="line-height: 1.8; font-size: 0.95em;">
                    • <b>{pct_1_gelas:.1f}%</b> minum <b>1 gelas kopi/hari</b><br>
                    • <b>{pct_tugas:.1f}%</b> minum saat <b>mengerjakan tugas</b><br>
                    • <b>{pct_cafe_favorit:.1f}%</b> tempat favorit <b>cafe/warkop</b><br>
                    • Pengeluaran median: <b>Rp {df_filtered['pengeluaran_num'].median() if 'pengeluaran_num' in df_filtered.columns and len(df_filtered) > 0 else 0:,.0f}/minggu</b><br>
                    • Frekuensi ke cafe: <b>1-2 kali/minggu</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="insight-box-roasted">
                <div class="insight-box-title-roasted">💭 ANALISIS 3: MAKNA NGOPI</div>
                <div style="line-height: 1.8; font-size: 0.95em;">
                    • <b>{pct_kebiasaan:.1f}%</b> sebagai <b>kebiasaan biasa</b><br>
                    • <b>{pct_kebutuhan:.1f}%</b> sebagai <b>kebutuhan wajib</b><br>
                    • <b>{pct_gaya_hidup:.1f}%</b> sebagai <b>gaya hidup</b><br>
                    • Hanya <b>3%</b> untuk <b>pergaulan</b><br>
                    • Fungsi utama: <b>dukung aktivitas belajar</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_anal2:
        st.markdown(f"""
            <div class="insight-box-roasted">
                <div class="insight-box-title-roasted">🔋 ANALISIS 2: EFEKTIVITAS KOPI</div>
                <div style="line-height: 1.8; font-size: 0.95em;">
                    • <b>{pct_tahan_ngantuk:.1f}%</b> efek <b>tahan ngantuk</b><br>
                    • <b>{pct_fokus_tinggi:.1f}%</b> fokus tingkat <b>tinggi</b><br>
                    • <b>{pct_durasi_3_4jam:.1f}%</b> durasi <b>3-4 jam</b><br>
                    • Efek <b>berbeda setiap individu</b><br>
                    • Hanya <b>{pct_lebih_fokus:.1f}%</b> merasa <b>lebih fokus</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="insight-box-roasted">
                <div class="insight-box-title-roasted">📈 ANALISIS 4: ANALISIS REGRESI</div>
                <div style="line-height: 1.8; font-size: 0.95em;">
                    • <b>Hubungan positif signifikan</b><br>
                    • <b>Fokus = 34.47 + 11.71 × Efek</b><br>
                    • <b>R² = {r2_value:.4f}</b> (<b>{r2_value*100:.1f}%</b> variasi)<br>
                    • <b>Korelasi Pearson: 0.4892</b><br>
                    • <b>P-value: < 0.001</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Kesimpulan Akhir
    if len(df_filtered) > 0:
        kesimpulan_text = f"""
            <div class="insight-box-dark" style="margin-top: 2rem;">
                <div class="insight-box-title">🎯 KESIMPULAN AKHIR</div>
                <div style="line-height: 1.9; font-size: 0.98em;">
                    Kopi berperan <b>penting</b> dalam mendukung aktivitas akademik mahasiswa dengan <b>pola konsumsi wajar</b> 
                    (<b>1 gelas/hari</b>) dan <b>pengeluaran terkendali (Rp {df_filtered['pengeluaran_num'].median() if 'pengeluaran_num' in df_filtered.columns else 0:,.0f}/minggu)</b>. Sebagian besar (<b>{pct_tugas:.1f}%</b>) minum kopi saat 
                    <b>mengerjakan tugas</b> dengan fungsi utama <b>membantu fokus belajar</b>. Analisis regresi membuktikan 
                    <b>pengaruh positif signifikan</b> efek kopi terhadap fokus (<b>11.71%</b> peningkatan per tingkat efek), 
                    meskipun hanya menjelaskan <b>23.9%</b> variasi fokus. <b>Respon terhadap kopi sangat bervariasi</b> antar 
                    individu, dengan efek yang dirasakan berbeda-beda, menunjukkan bahwa <b>efektivitasnya bergantung</b> pada kondisi 
                    fisiologis dan psikologis masing-masing mahasiswa.
                </div>
            </div>
        """
    else:
        kesimpulan_text = """
            <div class="insight-box-dark" style="margin-top: 2rem;">
                <div class="insight-box-title">🎯 KESIMPULAN AKHIR</div>
                <div style="line-height: 1.9;">
                    Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter untuk melihat analisis.
                </div>
            </div>
        """
    st.markdown(kesimpulan_text, unsafe_allow_html=True)

# ==================== TAB 2: POLA KONSUMSI ====================
with tab2:
    st.markdown('<div class="section-title">☕ ANALISIS 1: POLA KONSUMSI DAN KEBIASAAN NGOPI</div>', unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.info("📊 Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter untuk melihat analisis.")
    else:
        # RINGKASAN ANALISIS (selalu tampil)
        st.markdown('<div class="subsection-title">📌 Ringkasan Analisis</div>', unsafe_allow_html=True)
        
        summary_points = []
        if 'gelas_num' in df_filtered.columns:
            gelas_mode = df_filtered['gelas_num'].mode()
            modus_gelas = gelas_mode[0] if len(gelas_mode) > 0 else 0
            count_1_gelas = (df_filtered['gelas_num'] == 1).sum()
            summary_points.append(f"- {count_1_gelas} mahasiswa minum 1 gelas kopi per hari (modus: {modus_gelas} gelas)")
        
        if 'waktu_ngopi' in df_filtered.columns:
            waktu_counts = safe_value_counts(df_filtered, 'waktu_ngopi')
            if len(waktu_counts) > 0:
                waktu_terbanyak = waktu_counts.index[0]
                pct_waktu = (waktu_counts.values[0] / len(df_filtered)) * 100
                summary_points.append(f"- {pct_waktu:.1f}% minum kopi saat {waktu_terbanyak.lower()}")
        
        if 'pengeluaran_num' in df_filtered.columns:
            median_pengeluaran = df_filtered['pengeluaran_num'].median()
            summary_points.append(f"- Median pengeluaran: Rp {median_pengeluaran:,.0f}/minggu")

        for point in summary_points:
            st.markdown(point)

        st.markdown("---")

        # CHART UTAMA (selalu tampil)
        st.markdown('<div class="section-title">🔎 Tren Minum Kopi</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-title">📈 Distribusi Konsumsi per Hari</div>', unsafe_allow_html=True)
            if 'gelas_num' in df_filtered.columns:
                gelas_dist = safe_value_counts(df_filtered, 'gelas_num').sort_index()
                if len(gelas_dist) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK DISTRIBUSI KONSUMSI
                    custom_palette_gelas = [
                        '#E1A95F',  # Burnt Yellow
                        '#C65A1E',  # Burnt Orange  
                        '#B87333',  # Copper
                        '#800000',  # Maroon
                        '#7D387D',  # Plum
                        '#3A66A7',  # Galaxy Blue
                        '#708238'   # Olive Green
                    ]
                    
                    bars = ax.bar(gelas_dist.index, gelas_dist.values, 
                                  color=custom_palette_gelas[:len(gelas_dist)],
                                  edgecolor='#3e2723', linewidth=2, alpha=0.9)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                               ha='center', va='bottom', fontweight='600', fontsize=13, color='#3e2723')
                    
                    ax.set_xlabel('Jumlah Gelas', fontsize=13, fontweight='600', color='#3e2723')
                    ax.set_ylabel('Responden', fontsize=13, fontweight='600', color='#3e2723')
                    ax.grid(axis='y', alpha=0.3, color='#BCAAA4', linestyle='--', linewidth=0.8)
                    ax.set_axisbelow(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    ax.spines['left'].set_linewidth(1.5)
                    ax.spines['bottom'].set_linewidth(1.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data konsumsi untuk ditampilkan")
        
        with col2:
            st.markdown('<div class="subsection-title">📍 Tempat Favorit Ngopi</div>', unsafe_allow_html=True)
            if 'tempat_favorit' in df_filtered.columns:
                tempat_dist = safe_value_counts(df_filtered, 'tempat_favorit')
                if len(tempat_dist) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK TEMPAT FAVORIT
                    custom_palette_tempat = [
                        '#E1A95F',  # Burnt Yellow
                        '#3A66A7',  # Galaxy Blue  
                        '#C65A1E',  # Burnt Orange
                        '#708238',  # Olive Green
                        '#B87333',  # Copper
                        '#C72C48',  # Raspberry
                        '#800000',  # Maroon
                        '#7D387D'   # Plum
                    ]
                    
                    colors = [custom_palette_tempat[i % len(custom_palette_tempat)] for i in range(len(tempat_dist))]
                    
                    bars = ax.barh(range(len(tempat_dist)), tempat_dist.values,
                                   color=colors,
                                   edgecolor='#3e2723', linewidth=2, alpha=0.9)
                    
                    ax.set_yticks(range(len(tempat_dist)))
                    ax.set_yticklabels(tempat_dist.index, fontsize=12, color='#3e2723', weight='500')
                    ax.set_xlabel('Responden', fontsize=13, fontweight='600', color='#3e2723')
                    
                    for i, (bar, v) in enumerate(zip(bars, tempat_dist.values)):
                        ax.text(v + 1, i, str(v), va='center', fontweight='600', fontsize=13, color='#3e2723')
                    
                    ax.grid(axis='x', alpha=0.3, color='#BCAAA4', linestyle='--', linewidth=0.8)
                    ax.set_axisbelow(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    ax.spines['left'].set_linewidth(1.5)
                    ax.spines['bottom'].set_linewidth(1.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data tempat untuk ditampilkan")

        # CHART DETAIL (hanya tampil jika show_detailed_charts = True)
        if show_detailed_charts:
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Distribusi Kebiasaan Ngopi</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<div class="subsection-title">⏰ Waktu Minum Kopi</div>', unsafe_allow_html=True)
                if 'waktu_ngopi' in df_filtered.columns:
                    waktu_dist = safe_value_counts(df_filtered, 'waktu_ngopi')
                    waktu_dist = waktu_dist[waktu_dist > 0]
                    
                    if len(waktu_dist) > 0:
                        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
                        
                        # PALETTE WARNA YANG LEBIH SEIMBANG - CAMPUR HANGAT & DINGIN
                        balanced_palette = [
                            '#E1A95F',  # Burnt Yellow - hangat (Saat mengerjakan tugas - 70.3%)
                            '#3A66A7',  # Galaxy Blue - dingin (Saat kuliah - 15.2%)
                            '#708238',  # Olive Green - netral (Sebelum berangkat kuliah - 12.7%)
                            '#C72C48',  # Raspberry - hangat (Setelah kuliah - 1.8%)
                            '#C65A1E',  # Burnt Orange - hangat (jika ada kategori lain)
                            '#7D387D'   # Plum - dingin (jika ada kategori lain)
                        ]
                        
                        wedges, texts, autotexts = ax.pie(waktu_dist.values, labels=waktu_dist.index, 
                                                          autopct='%1.1f%%', 
                                                          colors=balanced_palette[:len(waktu_dist)],
                                                          startangle=90, textprops={'fontsize': 11, 'color': '#3e2723'})
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('700')
                            autotext.set_fontsize(10)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak ada data waktu untuk ditampilkan")
            
            with col4:
                st.markdown('<div class="subsection-title">🛍️ Cara Mendapatkan Kopi</div>', unsafe_allow_html=True)
                if 'cara_dapat_kopi' in df_filtered.columns:
                    cara_dist = safe_value_counts(df_filtered, 'cara_dapat_kopi')
                    cara_dist = cara_dist[cara_dist > 0]
                    
                    if len(cara_dist) > 0:
                        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
                        
                        # PALETTE WARNA CUSTOM UNTUK PIE CHART CARA
                        custom_palette_cara = [
                            '#3A66A7',  # Galaxy Blue
                            '#708238',  # Olive Green  
                            '#E1A95F',  # Burnt Yellow
                            '#C72C48',  # Raspberry
                            '#7D387D',  # Plum
                            '#C65A1E',  # Burnt Orange
                            '#B87333',  # Copper
                            '#800000'   # Maroon
                        ]
                        
                        wedges, texts, autotexts = ax.pie(cara_dist.values, labels=cara_dist.index,
                                                          autopct='%1.1f%%',
                                                          colors=custom_palette_cara[:len(cara_dist)],
                                                          startangle=45, textprops={'fontsize': 11, 'color': '#3e2723'})
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('700')
                            autotext.set_fontsize(10)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak ada data cara untuk ditampilkan")

            # DIAGRAM BATANG FREKUENSI NONGKRONG SAJA
            st.markdown("---")
            st.markdown('<div class="section-title">🏪 Frekuensi Nongkrong di Cafe</div>', unsafe_allow_html=True)
            
            if 'frekuensi_cafe_num' in df_filtered.columns:
                # Gunakan data numerik yang sudah ada
                cafe_data = df_filtered['frekuensi_cafe_num']
                
                # Filter yang pernah nongkrong (>0)
                cafe_active = cafe_data[cafe_data > 0]
                
                if len(cafe_active) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # Buat diagram batang sederhana
                    value_counts = cafe_active.value_counts().sort_index()
                    
                    # PALETTE WARNA CUSTOM UNTUK FREKUENSI CAFE
                    custom_palette_cafe = [
                        '#E1A95F',  # Burnt Yellow
                        '#C65A1E',  # Burnt Orange
                        '#B87333',  # Copper
                        '#800000',  # Maroon
                        '#C72C48',  # Raspberry
                        '#7D387D',  # Plum
                        '#3A66A7',  # Galaxy Blue
                        '#708238'   # Olive Green
                    ]
                    
                    bars = ax.bar([str(int(x)) for x in value_counts.index], value_counts.values,
                                  color=custom_palette_cafe[:len(value_counts)],
                                  edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                               ha='center', va='bottom', fontweight='600', fontsize=13, color='#3e2723')
                    
                    ax.set_xlabel('Frekuensi per Minggu', fontsize=13, fontweight='600', color='#3e2723')
                    ax.set_ylabel('Jumlah Responden', fontsize=13, fontweight='600', color='#3e2723')
                    ax.grid(axis='y', alpha=0.2, color='#6d4c41')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada responden yang nongkrong di cafe")
            else:
                st.info("📊 Kolom frekuensi_cafe_num tidak ditemukan")

        # STATISTIK DETAIL (hanya tampil jika show_statistics = True)
        if show_statistics:
            st.markdown("---")
            st.markdown('<div class="section-title">📈 Statistik Detail Konsumsi</div>', unsafe_allow_html=True)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                if 'gelas_num' in df_filtered.columns:
                    st.metric("Mean Konsumsi", f"{df_filtered['gelas_num'].mean():.2f} gelas/hari")
                    st.metric("Median Konsumsi", f"{df_filtered['gelas_num'].median():.1f} gelas/hari")
                    mode_gelas = df_filtered['gelas_num'].mode()
                    st.metric("Modus Konsumsi", f"{mode_gelas[0] if len(mode_gelas) > 0 else 0} gelas/hari")
            
            with col_stat2:
                if 'gelas_num' in df_filtered.columns:
                    st.metric("Std Dev Konsumsi", f"{df_filtered['gelas_num'].std():.2f}")
                    st.metric("Variansi", f"{df_filtered['gelas_num'].var():.2f}")
                    st.metric("Range", f"{df_filtered['gelas_num'].max() - df_filtered['gelas_num'].min():.1f}")
            
            with col_stat3:
                if 'gelas_num' in df_filtered.columns:
                    st.metric("Konsumsi Min", f"{df_filtered['gelas_num'].min():.0f} gelas")
                    st.metric("Konsumsi Max", f"{df_filtered['gelas_num'].max():.0f} gelas")
                    st.metric("Q1 (25%)", f"{df_filtered['gelas_num'].quantile(0.25):.1f} gelas")

        st.markdown("---")

        # Analisis Pengeluaran
        st.markdown('<div class="section-title">💰 Analisis Pengeluaran</div>', unsafe_allow_html=True)
        
        col_peng1, col_peng2 = st.columns(2)
        
        with col_peng1:
            if 'pengeluaran_num' in df_filtered.columns:
                st.markdown('<div class="subsection-title">🧮 Distribusi Pengeluaran</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                
                # WARNA CUSTOM UNTUK HISTOGRAM PENGELUARAN
                ax.hist(df_filtered['pengeluaran_num'], bins=10, color='#B87333', alpha=0.7, edgecolor='#3e2723', linewidth=1.5)
                
                ax.set_xlabel('Pengeluaran per Minggu (Rp)', fontsize=13, fontweight='600', color='#3e2723')
                ax.set_ylabel('Frekuensi', fontsize=13, fontweight='600', color='#3e2723')
                ax.grid(axis='y', alpha=0.2, color='#6d4c41')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#6d4c41')
                ax.spines['bottom'].set_color('#6d4c41')
                plt.tight_layout()
                st.pyplot(fig)
        
        with col_peng2:
            st.markdown('<div class="subsection-title">🧾 Statistik Pengeluaran</div>', unsafe_allow_html=True)
            if 'pengeluaran_num' in df_filtered.columns:
                mean_pengeluaran = df_filtered['pengeluaran_num'].mean()
                median_pengeluaran = df_filtered['pengeluaran_num'].median()
                min_pengeluaran = df_filtered['pengeluaran_num'].min()
                max_pengeluaran = df_filtered['pengeluaran_num'].max()
                
                st.metric("Median Pengeluaran", f"Rp {median_pengeluaran:,.0f}/minggu")
                st.metric("Rata-rata Pengeluaran", f"Rp {mean_pengeluaran:,.0f}/minggu")
                st.metric("Pengeluaran Minimum", f"Rp {min_pengeluaran:,.0f}/minggu")
                st.metric("Pengeluaran Maksimum", f"Rp {max_pengeluaran:,.0f}/minggu")
                
                # STATISTIK DETAIL PENGELUARAN
                if show_statistics:
                    st.metric("Std Dev Pengeluaran", f"Rp {df_filtered['pengeluaran_num'].std():,.0f}")
                    st.metric("Q1 (25%)", f"Rp {df_filtered['pengeluaran_num'].quantile(0.25):,.0f}")
                    st.metric("Q3 (75%)", f"Rp {df_filtered['pengeluaran_num'].quantile(0.75):,.0f}")

# ==================== TAB 3: EFEKTIVITAS KOPI ====================
with tab3:
    st.markdown('<div class="section-title">🔋 ANALISIS 2: PERAN KOPI TERHADAP FOKUS DAN DURASI BELAJAR</div>', unsafe_allow_html=True)

    if len(df_filtered) == 0:
        st.info("📊 Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter untuk melihat analisis.")
    else:
        st.markdown('<div class="subsection-title">📌 Ringkasan Analisis</div>', unsafe_allow_html=True)
        
        summary_points = []
        
        if 'durasi_num' in df_filtered.columns:
            avg_durasi = df_filtered['durasi_num'].mean()
            median_durasi = df_filtered['durasi_num'].median()
            summary_points.append(f"- Durasi belajar: rata-rata {avg_durasi:.1f} jam (median: {median_durasi:.1f} jam)")
        
        if 'gelas_num' in df_filtered.columns and 'fokus_num' in df_filtered.columns:
            valid_data = df_filtered[['gelas_num', 'fokus_num']].dropna()
            if len(valid_data) > 1:
                corr_fokus = valid_data.corr().iloc[0, 1]
                summary_points.append(f"- Korelasi konsumsi-fokus: {corr_fokus:.3f}")
        
        if 'gelas_num' in df_filtered.columns and 'durasi_num' in df_filtered.columns:
            valid_data = df_filtered[['gelas_num', 'durasi_num']].dropna()
            if len(valid_data) > 1:
                corr_durasi = valid_data.corr().iloc[0, 1]
                summary_points.append(f"- Korelasi konsumsi-durasi: {corr_durasi:.3f}")
        
        # Rata-rata tingkat fokus berdasarkan efek kopi
        if 'efek_kopi' in df_filtered.columns and 'fokus_num' in df_filtered.columns:
            efek_counts = safe_value_counts(df_filtered, 'efek_kopi')
            if len(efek_counts) > 0:
                efek_terbanyak = efek_counts.index[0]
                jumlah_sampel = efek_counts.iloc[0]
                rata_fokus = df_filtered[df_filtered['efek_kopi'] == efek_terbanyak]['fokus_num'].mean()
                summary_points.append(f"- Rata-rata tingkat fokus: {rata_fokus:.1f}% pada mahasiswa yang merasakan '{efek_terbanyak}' ({jumlah_sampel} responden)")

        for point in summary_points:
            st.markdown(point)

        st.markdown("---")

        # 1. 🔍 DETAIL ANALISIS (selalu tampil)
        st.markdown('<div class="section-title">🔍 Detail Analisis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="subsection-title">🧠 Detail Persepsi Fokus</div>', unsafe_allow_html=True)
            if 'kat_fokus' in df_filtered.columns:
                fokus_order = ['Tidak membantu', 'Rendah (1-30%)', 'Sedang (31-50%)', 'Tinggi (51-70%)', 'Sangat Tinggi (>70%)']
                fokus_dist = safe_value_counts(df_filtered, 'kat_fokus').reindex(fokus_order)
                fokus_dist = fokus_dist.dropna()
                
                if len(fokus_dist) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK FOKUS
                    custom_palette_fokus = [
                        '#E1A95F',  # Burnt Yellow
                        '#C65A1E',  # Burnt Orange
                        '#B87333',  # Copper
                        '#800000',  # Maroon
                        '#7D387D'   # Plum
                    ]
                    
                    bars = ax.barh(range(len(fokus_dist)), fokus_dist.values,
                                   color=custom_palette_fokus[:len(fokus_dist)],  # <- DIUBAH DI SINI
                                   edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                    
                    ax.set_yticks(range(len(fokus_dist)))
                    ax.set_yticklabels(fokus_dist.index, fontsize=11, color='#3e2723')
                    ax.set_xlabel('Jumlah Responden', fontsize=13, fontweight='600', color='#3e2723')
                    
                    for i, (bar, v) in enumerate(zip(bars, fokus_dist.values)):
                        if pd.notna(v):
                            ax.text(v + 1, i, f'{int(v)}', va='center', fontweight='600', fontsize=11, color='#3e2723')
                    
                    ax.grid(axis='x', alpha=0.2, color='#6d4c41')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data fokus untuk ditampilkan")

        with col2:
            st.markdown('<div class="subsection-title">⏱️ Durasi Tahan Belajar</div>', unsafe_allow_html=True)
            if 'kat_durasi' in df_filtered.columns:
                durasi_order = ['Tidak berpengaruh', '1-2 jam', '3-4 jam', '5-6 jam', '> 6 jam']
                durasi_dist = safe_value_counts(df_filtered, 'kat_durasi').reindex(durasi_order)
                durasi_dist = durasi_dist.dropna()
                
                if len(durasi_dist) > 0:
                    # PALETTE WARNA CUSTOM UNTUK DURASI
                    custom_palette_durasi = [
                        '#3A66A7',  # Galaxy Blue
                        '#708238',  # Olive Green
                        '#E1A95F',  # Burnt Yellow
                        '#C65A1E',  # Burnt Orange
                        '#C72C48'   # Raspberry
                    ]
                    
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    bars = ax.bar(range(len(durasi_dist)), durasi_dist.values,
                                  color=custom_palette_durasi[:len(durasi_dist)],  # <- DIUBAH DI SINI
                                  edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                    
                    ax.set_xticks(range(len(durasi_dist)))
                    ax.set_xticklabels(durasi_dist.index, rotation=45, ha='right', fontsize=11, color='#3e2723')
                    ax.set_ylabel('Jumlah Responden', fontsize=13, fontweight='600', color='#3e2723')
                    
                    for i, (bar, v) in enumerate(zip(bars, durasi_dist.values)):
                        if pd.notna(v):
                            ax.text(bar.get_x() + bar.get_width()/2., v, f'{int(v)}', 
                                   ha='center', va='bottom', fontweight='600', fontsize=11, color='#3e2723')
                    
                    ax.grid(axis='y', alpha=0.2, color='#6d4c41')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data durasi untuk ditampilkan")

        # 2. 📈 ANALISIS KORELASI (hanya detail)
        if show_detailed_charts:
            st.markdown("---")
            st.markdown('<div class="section-title">📈 Analisis Korelasi</div>', unsafe_allow_html=True)

            if 'gelas_num' in df_filtered.columns and 'fokus_num' in df_filtered.columns and 'durasi_num' in df_filtered.columns:
                # Hitung korelasi dengan data yang valid
                valid_fokus = df_filtered[['gelas_num', 'fokus_num']].dropna()
                valid_durasi = df_filtered[['gelas_num', 'durasi_num']].dropna()
                
                if len(valid_fokus) > 1:
                    corr_fokus = valid_fokus.corr().iloc[0, 1]
                else:
                    corr_fokus = 0
                    
                if len(valid_durasi) > 1:
                    corr_durasi = valid_durasi.corr().iloc[0, 1]
                else:
                    corr_durasi = 0
                
                col_scatter1, col_scatter2 = st.columns(2)
                
                with col_scatter1:
                    st.markdown('<div class="subsection-title">Hubungan Konsumsi Kopi dengan Tingkat Fokus</div>', unsafe_allow_html=True)
                    
                    if len(valid_fokus) > 1:
                        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
                        
                        # Scatter plot konsumsi vs fokus dengan warna custom
                        scatter = ax.scatter(valid_fokus['gelas_num'], valid_fokus['fokus_num'], 
                                           alpha=0.7, s=60, color='#B87333', edgecolors='#3e2723', linewidth=1)  # <- DIUBAH DI SINI
                        
                        # Garis regresi dengan warna custom
                        x_line = np.linspace(valid_fokus['gelas_num'].min(), valid_fokus['gelas_num'].max(), 100)
                        z = np.polyfit(valid_fokus['gelas_num'], valid_fokus['fokus_num'], 1)
                        p = np.poly1d(z)
                        ax.plot(x_line, p(x_line), '#800000', linewidth=2, alpha=0.8)  # <- DIUBAH DI SINI
                        
                        ax.set_xlabel('Jumlah Gelas per Hari', fontsize=11, fontweight='600', color='#3e2723')
                        ax.set_ylabel('Tingkat Fokus (%)', fontsize=11, fontweight='600', color='#3e2723')
                        
                        ax.grid(alpha=0.2, color='#6d4c41')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#6d4c41')
                        ax.spines['bottom'].set_color('#6d4c41')
                        
                        # Hanya korelasi di kanan atas
                        ax.text(0.95, 0.95, f'r = {corr_fokus:.3f}', transform=ax.transAxes, fontsize=11,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.9, edgecolor='#6d4c41'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak cukup data untuk scatter plot (minimal 2 data point diperlukan)")
                
                with col_scatter2:
                    st.markdown('<div class="subsection-title">Hubungan Konsumsi Kopi dengan Durasi Belajar</div>', unsafe_allow_html=True)
                    
                    if len(valid_durasi) > 1:
                        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
                        
                        # Scatter plot konsumsi vs durasi dengan warna custom
                        scatter = ax.scatter(valid_durasi['gelas_num'], valid_durasi['durasi_num'], 
                                           alpha=0.7, s=60, color='#3A66A7', edgecolors='#3e2723', linewidth=1)  # <- DIUBAH DI SINI
                        
                        # Garis regresi dengan warna custom
                        x_line = np.linspace(valid_durasi['gelas_num'].min(), valid_durasi['gelas_num'].max(), 100)
                        z = np.polyfit(valid_durasi['gelas_num'], valid_durasi['durasi_num'], 1)
                        p = np.poly1d(z)
                        ax.plot(x_line, p(x_line), '#708238', linewidth=2, alpha=0.8)  # <- DIUBAH DI SINI
                        
                        ax.set_xlabel('Jumlah Gelas per Hari', fontsize=11, fontweight='600', color='#3e2723')
                        ax.set_ylabel('Durasi Belajar (Jam)', fontsize=11, fontweight='600', color='#3e2723')
                        
                        ax.grid(alpha=0.2, color='#6d4c41')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#6d4c41')
                        ax.spines['bottom'].set_color('#6d4c41')
                        
                        # Hanya korelasi di kanan atas
                        ax.text(0.95, 0.95, f'r = {corr_durasi:.3f}', transform=ax.transAxes, fontsize=11,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.9, edgecolor='#6d4c41'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak cukup data untuk scatter plot (minimal 2 data point diperlukan)")

            else:
                st.info("📊 Tidak cukup data untuk analisis scatter plot")

        # 3. 📊 DISTRIBUSI PERSEPSI DAN EFEK KOPI (hanya detail)
        if show_detailed_charts:
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Distribusi Persepsi dan Efek Kopi</div>', unsafe_allow_html=True)
            col_pie1, col_pie2 = st.columns(2)

            with col_pie1:
                st.markdown('<div class="subsection-title">🧠 Persepsi Peningkatan Fokus</div>', unsafe_allow_html=True)
                if 'kat_fokus' in df_filtered.columns:
                    fokus_order = ['Tidak membantu', 'Rendah (1-30%)', 'Sedang (31-50%)', 'Tinggi (51-70%)', 'Sangat Tinggi (>70%)']
                    fokus_dist = safe_value_counts(df_filtered, 'kat_fokus').reindex(fokus_order)
                    fokus_dist = fokus_dist.dropna()
                    fokus_dist = fokus_dist[fokus_dist > 0]
                    
                    if len(fokus_dist) > 0:
                        # PALETTE WARNA CUSTOM UNTUK PIE CHART FOKUS
                        custom_palette_pie_fokus = [
                            '#E1A95F',  # Burnt Yellow
                            '#C65A1E',  # Burnt Orange
                            '#B87333',  # Copper
                            '#800000',  # Maroon
                            '#7D387D'   # Plum
                        ]
                        
                        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
                        wedges, texts, autotexts = ax.pie(fokus_dist.values, labels=fokus_dist.index,
                                                         autopct='%1.1f%%', colors=custom_palette_pie_fokus[:len(fokus_dist)],  # <- DIUBAH DI SINI
                                                         startangle=90, textprops={'fontsize': 10, 'color': '#3e2723'})
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('700')
                            autotext.set_fontsize(10)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak ada data fokus untuk ditampilkan")

            with col_pie2:
                st.markdown('<div class="subsection-title">💥 Efek yang Dirasakan</div>', unsafe_allow_html=True)
                if 'efek_kopi' in df_filtered.columns:
                    efek_dist = safe_value_counts(df_filtered, 'efek_kopi')
                    efek_dist = efek_dist[efek_dist > 0]
                    
                    if len(efek_dist) > 0:
                        # PALETTE WARNA CUSTOM UNTUK PIE CHART EFEK
                        custom_palette_pie_efek = [
                            '#3A66A7',  # Galaxy Blue
                            '#708238',  # Olive Green
                            '#E1A95F',  # Burnt Yellow
                            '#C72C48',  # Raspberry
                            '#7D387D'   # Plum
                        ]
                        
                        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
                        wedges, texts, autotexts = ax.pie(efek_dist.values, labels=efek_dist.index,
                                                         autopct='%1.1f%%', colors=custom_palette_pie_efek[:len(efek_dist)],  # <- DIUBAH DI SINI
                                                         startangle=90, textprops={'fontsize': 10, 'color': '#3e2723'})
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('700')
                            autotext.set_fontsize(10)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak ada data efek untuk ditampilkan")

        # 4. 🧮 RATA-RATA FOKUS PER KATEGORI EFEK KOPI (selalu tampil)
        st.markdown("---")
        st.markdown('<div class="section-title">🧮 Rata-rata Fokus per Kategori Efek Kopi</div>', unsafe_allow_html=True)

        urutan_manual = ['Lebih fokus', 'Lebih semangat', 'Tahan ngantuk', 'Biasa aja', 'Jadi cemas']
        if 'efek_kopi' in df_filtered.columns and 'fokus_num' in df_filtered.columns and len(df_filtered) > 0:
            kategori_analisis = safe_groupby(df_filtered, 'efek_kopi', 'fokus_num', 'mean')
            if len(kategori_analisis) > 0:
                kategori_analisis = kategori_analisis.reindex(urutan_manual)
                kategori_analisis = kategori_analisis.dropna()

                if len(kategori_analisis) > 0:
                    counts = safe_groupby(df_filtered, 'efek_kopi', 'fokus_num', 'count')
                    stds = df_filtered.groupby('efek_kopi')['fokus_num'].std()
                    
                    col_kat1, col_kat2 = st.columns([2, 1])
                    
                    with col_kat1:
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                        
                        # PALETTE WARNA CUSTOM UNTUK RATA-RATA FOKUS
                        custom_palette_rata_fokus = [
                            '#E1A95F',  # Burnt Yellow
                            '#C65A1E',  # Burnt Orange
                            '#B87333',  # Copper
                            '#800000',  # Maroon
                            '#7D387D'   # Plum
                        ]
                        
                        bars = ax.barh(range(len(kategori_analisis)), kategori_analisis.values,
                                    color=custom_palette_rata_fokus[:len(kategori_analisis)],  # <- DIUBAH DI SINI
                                    edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                        
                        ax.set_yticks(range(len(kategori_analisis)))
                        ax.set_yticklabels(kategori_analisis.index, fontsize=10, color='#3e2723')
                        ax.set_xlabel('Rata-rata Fokus (%)', fontsize=11, fontweight='600', color='#3e2723')
                        
                        for i, (bar, v) in enumerate(zip(bars, kategori_analisis.values)):
                            if pd.notna(v):
                                count_val = counts.get(kategori_analisis.index[i], 0)
                                std_val = stds.get(kategori_analisis.index[i], 0)
                                ax.text(v + 2, i, f'{v:.1f}% (n={int(count_val)})', va='center', fontweight='600', fontsize=9, color='#3e2723')
                        
                        ax.grid(axis='x', alpha=0.2, color='#6d4c41')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#6d4c41')
                        ax.spines['bottom'].set_color('#6d4c41')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col_kat2:
                        st.markdown('<div class="subsection-title">Interpretasi:</div>', unsafe_allow_html=True)
                        for efek, row in kategori_analisis.iloc[::-1].items():
                            count_val = counts.get(efek, 0)
                            std_val = stds.get(efek, 0)
                            st.markdown(f"- **{efek}**: {row:.1f}% ±{std_val:.1f} (n={int(count_val)})")
                else:
                    st.info("📊 Tidak ada data kategori untuk ditampilkan")
            else:
                st.info("📊 Tidak ada data yang cukup untuk analisis kategori efek kopi")
        else:
            st.info("📊 Tidak ada data yang cukup untuk analisis kategori efek kopi")

        # 5. 🔗 HUBUNGAN EFEK KOPI DENGAN DURASI BELAJAR (selalu tampil)
        st.markdown("---")
        st.markdown('<div class="section-title">🔗 Hubungan Efek Kopi dengan Durasi Belajar</div>', unsafe_allow_html=True)
        
        if 'efek_kopi' in df_filtered.columns and 'durasi_num' in df_filtered.columns:
            efek_durasi = safe_groupby(df_filtered, 'efek_kopi', 'durasi_num', 'mean')
            if len(efek_durasi) > 0:
                efek_durasi = efek_durasi.dropna()
                
                if len(efek_durasi) > 0:
                    counts = safe_groupby(df_filtered, 'efek_kopi', 'durasi_num', 'count')
                    
                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK HUBUNGAN EFEK-DURASI
                    custom_palette_efek_durasi = [
                        '#3A66A7',  # Galaxy Blue
                        '#708238',  # Olive Green
                        '#E1A95F',  # Burnt Yellow
                        '#C72C48',  # Raspberry
                        '#7D387D',  # Plum
                        '#C65A1E',  # Burnt Orange
                        '#B87333'   # Copper
                    ]
                    
                    bars = ax.bar(range(len(efek_durasi)), efek_durasi.values,
                                  color=custom_palette_efek_durasi[:len(efek_durasi)],  # <- DIUBAH DI SINI
                                  edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                    
                    ax.set_xticks(range(len(efek_durasi)))
                    ax.set_xticklabels(efek_durasi.index, rotation=30, ha='right', fontsize=12, color='#3e2723')
                    ax.set_ylabel('Rata-rata Durasi (Jam)', fontsize=13, fontweight='600', color='#3e2723')
                    
                    for i, (bar, count) in enumerate(zip(bars, efek_durasi.index)):
                        height = bar.get_height()
                        if pd.notna(height):
                            count_val = counts.get(count, 0)
                            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}h\n(n={int(count_val)})',
                                   ha='center', va='bottom', fontweight='600', fontsize=11, color='#3e2723')
                    
                    ax.grid(axis='y', alpha=0.2, color='#6d4c41')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data hubungan efek-durasi untuk ditampilkan")
            else:
                st.info("📊 Tidak ada data hubungan efek-durasi untuk ditampilkan")

        # 6. 🕒 DISTRIBUSI DURASI BELAJAR (selalu tampil)
        st.markdown("---")
        st.markdown('<div class="section-title">🕒 Distribusi Durasi Belajar</div>', unsafe_allow_html=True)
        
        col_dur1, col_dur2 = st.columns(2)
        
        with col_dur1:
            st.markdown('<div class="subsection-title">🌀 Pola Durasi Belajar</div>', unsafe_allow_html=True)
            if 'kat_durasi' in df_filtered.columns:
                durasi_order = ['Tidak berpengaruh', '1-2 jam', '3-4 jam', '5-6 jam', '> 6 jam']
                durasi_dist = safe_value_counts(df_filtered, 'kat_durasi').reindex(durasi_order)
                durasi_dist = durasi_dist.dropna()
                durasi_dist = durasi_dist[durasi_dist > 0]
                
                if len(durasi_dist) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK PIE DURASI
                    custom_palette_pie_durasi = [
                        '#E1A95F',  # Burnt Yellow
                        '#C65A1E',  # Burnt Orange
                        '#B87333',  # Copper
                        '#800000',  # Maroon
                        '#7D387D'   # Plum
                    ]
                    
                    wedges, texts, autotexts = ax.pie(durasi_dist.values, labels=durasi_dist.index,
                                                     autopct='%1.1f%%', colors=custom_palette_pie_durasi[:len(durasi_dist)],  # <- DIUBAH DI SINI
                                                     startangle=90, textprops={'fontsize': 8, 'color': '#3e2723'})
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('700')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data durasi untuk ditampilkan")
        
        with col_dur2:
            st.markdown('<div class="subsection-title">📋 Statistik Tambahan</div>', unsafe_allow_html=True)
            if 'durasi_num' in df_filtered.columns:
                durasi_min = df_filtered['durasi_num'].min()
                durasi_max = df_filtered['durasi_num'].max()
                durasi_median = df_filtered['durasi_num'].median()
                st.metric("Durasi Minimum", f"{durasi_min:.1f} jam")
                st.metric("Durasi Maksimum", f"{durasi_max:.1f} jam")
                st.metric("Durasi Median", f"{durasi_median:.1f} jam")

        # STATISTIK DETAIL (hanya tampil jika show_statistics = True)
        if show_statistics:
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Statistik Detail Fokus & Durasi</div>', unsafe_allow_html=True)
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                if 'fokus_num' in df_filtered.columns:
                    st.metric("Mean Fokus", f"{df_filtered['fokus_num'].mean():.1f}%")
                    st.metric("Median Fokus", f"{df_filtered['fokus_num'].median():.1f}%")
                    st.metric("Std Dev Fokus", f"{df_filtered['fokus_num'].std():.1f}%")
                    st.metric("Variansi Fokus", f"{df_filtered['fokus_num'].var():.1f}")
            
            with col_stat2:
                if 'durasi_num' in df_filtered.columns:
                    st.metric("Mean Durasi", f"{df_filtered['durasi_num'].mean():.1f} jam")
                    st.metric("Median Durasi", f"{df_filtered['durasi_num'].median():.1f} jam")
                    st.metric("Std Dev Durasi", f"{df_filtered['durasi_num'].std():.1f} jam")
                    st.metric("Variansi Durasi", f"{df_filtered['durasi_num'].var():.1f}")

# ==================== TAB 4: MAKNA NGOPI ====================
with tab4:
    st.markdown('<div class="section-title">💭 ANALISIS 3: MAKNA DAN FUNGSI NGOPI</div>', unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.info("📊 Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter untuk melihat analisis.")
    else:
        st.markdown('<div class="subsection-title">📌 Ringkasan Analisis</div>', unsafe_allow_html=True)
        
        summary_points = []
        
        if 'makna_ngopi' in df_filtered.columns and 'gelas_num' in df_filtered.columns:
            makna_konsumsi = safe_groupby(df_filtered, 'makna_ngopi', 'gelas_num', 'mean')
            if len(makna_konsumsi) > 0:
                makna_konsumsi_tertinggi = makna_konsumsi.idxmax()
                rata_konsumsi = makna_konsumsi.max()
                jumlah_responden = safe_groupby(df_filtered, 'makna_ngopi', 'gelas_num', 'count').get(makna_konsumsi_tertinggi, 0)
                summary_points.append(f"- Makna '{makna_konsumsi_tertinggi}' memiliki konsumsi tertinggi: {rata_konsumsi:.1f} gelas/hari ({jumlah_responden} responden)")

        if 'makna_ngopi' in df_filtered.columns:
            makna_counts = safe_value_counts(df_filtered, 'makna_ngopi')
            if len(makna_counts) > 0:
                makna_terbanyak = makna_counts.index[0]
                pct_makna = makna_counts.values[0] / len(df_filtered) * 100
                summary_points.append(f"- {pct_makna:.1f}% mahasiswa menganggap ngopi sebagai {makna_terbanyak.lower()}")

        if 'makna_ngopi' in df_filtered.columns:
            kebutuhan_count = (df_filtered['makna_ngopi'] == 'Kebutuhan wajib menemani mengerjakan tugas').sum()
            pct_kebutuhan = kebutuhan_count / len(df_filtered) * 100
            summary_points.append(f"- {pct_kebutuhan:.1f}% menganggapnya sebagai kebutuhan wajib saat mengerjakan tugas")

        for point in summary_points:
            st.markdown(point)

        st.markdown("---")

        # JUDUL BARU: MAKNA DALAM KONSUMSI KOPI
        st.markdown('<div class="section-title">☕ Makna dalam Konsumsi Kopi</div>', unsafe_allow_html=True)
        
        # Charts utama
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-title">💡 Persepsi Makna Ngopi</div>', unsafe_allow_html=True)
            if 'makna_ngopi' in df_filtered.columns:
                makna_dist = safe_value_counts(df_filtered, 'makna_ngopi')
                if len(makna_dist) > 0:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK MAKNA NGOPI
                    custom_palette_makna = [
                        '#E1A95F',  # Burnt Yellow
                        '#C65A1E',  # Burnt Orange
                        '#B87333',  # Copper
                        '#800000',  # Maroon
                        '#7D387D',  # Plum
                        '#3A66A7',  # Galaxy Blue
                        '#708238'   # Olive Green
                    ]
                    
                    bars = ax.barh(range(len(makna_dist)), makna_dist.values,
                                   color=custom_palette_makna[:len(makna_dist)],  # <- DIUBAH DI SINI
                                   edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                    
                    ax.set_yticks(range(len(makna_dist)))
                    ax.set_yticklabels(makna_dist.index, fontsize=11, color='#3e2723')
                    ax.set_xlabel('Jumlah Responden', fontsize=13, fontweight='600', color='#3e2723')
                    ax.set_ylabel('Makna Ngopi', fontsize=13, fontweight='600', color='#3e2723')
                    
                    for i, (bar, v) in enumerate(zip(bars, makna_dist.values)):
                        ax.text(v + 1, i, str(v), va='center', fontweight='600', color='#3e2723', fontsize=12)
                    
                    ax.grid(axis='x', alpha=0.2, color='#6d4c41')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data makna untuk ditampilkan")
        
        with col2:
            st.markdown('<div class="subsection-title">🔄 Hubungan Makna dengan Konsumsi</div>', unsafe_allow_html=True)
            if 'makna_ngopi' in df_filtered.columns and 'gelas_num' in df_filtered.columns:
                makna_konsumsi = safe_groupby(df_filtered, 'makna_ngopi', 'gelas_num', 'mean')
                if len(makna_konsumsi) > 0:
                    makna_konsumsi = makna_konsumsi.dropna()
                    
                    if len(makna_konsumsi) > 0:
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                        
                        # PALETTE WARNA CUSTOM UNTUK HUBUNGAN MAKNA-KONSUMSI
                        custom_palette_makna_konsumsi = [
                            '#3A66A7',  # Galaxy Blue
                            '#708238',  # Olive Green
                            '#E1A95F',  # Burnt Yellow
                            '#C72C48',  # Raspberry
                            '#7D387D',  # Plum
                            '#C65A1E'   # Burnt Orange
                        ]
                        
                        bars = ax.bar(range(len(makna_konsumsi)), makna_konsumsi.values,
                                      color=custom_palette_makna_konsumsi[:len(makna_konsumsi)],  # <- DIUBAH DI SINI
                                      edgecolor='#3e2723', linewidth=1.5, alpha=0.85)
                        
                        ax.set_xticks(range(len(makna_konsumsi)))
                        ax.set_xticklabels(makna_konsumsi.index, rotation=45, ha='right', fontsize=11, color='#3e2723')
                        ax.set_ylabel('Rata-rata Gelas/Hari', fontsize=13, fontweight='600', color='#3e2723')
                        ax.set_xlabel('Makna Ngopi', fontsize=13, fontweight='600', color='#3e2723')
                        
                        counts = safe_groupby(df_filtered, 'makna_ngopi', 'gelas_num', 'count')
                        for i, (bar, count) in enumerate(zip(bars, makna_konsumsi.index)):
                            height = bar.get_height()
                            count_val = counts.get(count, 0)
                            if pd.notna(height) and pd.notna(count_val):
                                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}\n(n={int(count_val)})',
                                       ha='center', va='bottom', fontweight='600', fontsize=11, color='#3e2723')
                        
                        ax.grid(axis='y', alpha=0.2, color='#6d4c41')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#6d4c41')
                        ax.spines['bottom'].set_color('#6d4c41')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak ada data hubungan makna-konsumsi untuk ditampilkan")
                else:
                    st.info("📊 Tidak ada data hubungan makna-konsumsi untuk ditampilkan")

        st.markdown("---")

        # Heatmap dan Pie Chart
        st.markdown('<div class="section-title">🔗 Analisis Hubungan Makna dengan Pengeluaran</div>', unsafe_allow_html=True)
        
        # Buat kategori pengeluaran
        if 'pengeluaran_num' in df_filtered.columns:
            try:
                bins = [0, 0.1, 15000, 30000, 50000, 100000, float('inf')]
                labels = ['Tidak mengeluarkan', '< Rp 15rb', 'Rp 15-30rb', 'Rp 30-50rb', 'Rp 50-100rb', '≥ Rp 100rb']
                df_filtered['kat_pengeluaran'] = pd.cut(df_filtered['pengeluaran_num'], bins=bins, labels=labels, right=False)
            except Exception:
                df_filtered['kat_pengeluaran'] = 'Tidak ada data'
        
        col_heat1, col_heat2 = st.columns(2)
        
        with col_heat1:
            st.markdown('<div class="subsection-title">💰 Heatmap: Makna vs Pengeluaran</div>', unsafe_allow_html=True)
            if 'makna_ngopi' in df_filtered.columns and 'kat_pengeluaran' in df_filtered.columns:
                try:
                    crosstab_pct = pd.crosstab(df_filtered['makna_ngopi'], df_filtered['kat_pengeluaran'], normalize='index') * 100
                    if len(crosstab_pct) > 0 and len(crosstab_pct.columns) > 0:
                        fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
                        
                        # HEATMAP KEMBALI KE WARNA BIASA (viridis)
                        sns.heatmap(crosstab_pct, annot=True, fmt='.1f', 
                                    cmap='viridis',  # <- DIUBAH DI SINI (kembali ke warna biasa)
                                    ax=ax, cbar_kws={'label': 'Persentase (%)'},
                                    linewidths=1, linecolor='#3e2723',
                                    annot_kws={'size': 9, 'weight': '600', 'color': 'white'})  # <- DIUBAH DI SINI
                        
                        ax.set_xlabel('Kategori Pengeluaran', fontsize=11, fontweight='600', color='#3e2723')
                        ax.set_ylabel('Makna Ngopi', fontsize=11, fontweight='600', color='#3e2723')
                        plt.xticks(rotation=45, ha='right', fontsize=9, color='#3e2723')
                        plt.yticks(rotation=0, fontsize=9, color='#3e2723')
                        
                        cbar = ax.collections[0].colorbar
                        cbar.ax.tick_params(labelsize=9)
                        cbar.set_label('Persentase (%)', fontsize=10, fontweight='600')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("📊 Tidak ada data heatmap untuk ditampilkan")
                except Exception:
                    st.info("📊 Tidak ada data heatmap untuk ditampilkan")
        
        with col_heat2:
            st.markdown('<div class="subsection-title">🌀 Distribusi Makna Ngopi</div>', unsafe_allow_html=True)
            if 'makna_ngopi' in df_filtered.columns:
                makna_dist = safe_value_counts(df_filtered, 'makna_ngopi')
                makna_dist = makna_dist[makna_dist > 0]
                
                if len(makna_dist) > 0:
                    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
                    
                    # PALETTE WARNA CUSTOM UNTUK PIE CHART MAKNA
                    custom_palette_pie_makna = [
                        '#E1A95F',  # Burnt Yellow
                        '#3A66A7',  # Galaxy Blue
                        '#C65A1E',  # Burnt Orange
                        '#708238',  # Olive Green
                        '#B87333',  # Copper
                        '#C72C48',  # Raspberry
                        '#7D387D'   # Plum
                    ]
                    
                    wedges, texts, autotexts = ax.pie(makna_dist.values, labels=makna_dist.index,
                                                     autopct='%1.1f%%', colors=custom_palette_pie_makna[:len(makna_dist)],  # <- DIUBAH DI SINI
                                                     startangle=90, textprops={'fontsize': 9, 'color': '#3e2723'})
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('700')
                        autotext.set_fontsize(9)
                    
                    for text in texts:
                        text.set_fontsize(9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 Tidak ada data makna untuk ditampilkan")

        st.markdown("---")

        # Tabel Distribusi
        st.markdown('<div class="section-title">📊 Distribusi Pengeluaran berdasarkan Makna Ngopi</div>', unsafe_allow_html=True)

        if 'makna_ngopi' in df_filtered.columns and 'kat_pengeluaran' in df_filtered.columns:
            st.markdown('<div class="subsection-title">📋 Jumlah Responden per Kategori</div>', unsafe_allow_html=True)
            crosstab_count = pd.crosstab(df_filtered['makna_ngopi'], df_filtered['kat_pengeluaran'])
            
            expected_columns = ['Tidak mengeluarkan', '< Rp 15rb', 'Rp 15-30rb', 'Rp 30-50rb', 'Rp 50-100rb', '≥ Rp 100rb']
            for col in expected_columns:
                if col not in crosstab_count.columns:
                    crosstab_count[col] = 0
            
            crosstab_count = crosstab_count[expected_columns]
            
            if len(crosstab_count) > 0 and len(crosstab_count.columns) > 0:
                crosstab_count['Total'] = crosstab_count.sum(axis=1)
                crosstab_count.loc['Total'] = crosstab_count.sum(axis=0)
                
                # TABEL DENGAN WARNA MAROON
                st.markdown(create_colored_table(
                    crosstab_count.reset_index(),
                    header_color='#800000',  # <- DIUBAH DI SINI (Maroon)
                    even_color='#f5f3f0', 
                    odd_color='#ffffff',
                    total_color='#5a0000'  # <- DIUBAH DI SINI (Maroon lebih gelap)
                ), unsafe_allow_html=True)
                
                st.markdown('<div class="subsection-title">📑 Persentase Distribusi per Makna Ngopi</div>', unsafe_allow_html=True)
                crosstab_pct_display = pd.crosstab(df_filtered['makna_ngopi'], df_filtered['kat_pengeluaran'], normalize='index') * 100
                
                for col in expected_columns:
                    if col not in crosstab_pct_display.columns:
                        crosstab_pct_display[col] = 0.0
                
                crosstab_pct_display = crosstab_pct_display[expected_columns]
                crosstab_pct_display = crosstab_pct_display.round(1)
                
                crosstab_pct_display_formatted = crosstab_pct_display.copy()
                for col in crosstab_pct_display_formatted.columns:
                    crosstab_pct_display_formatted[col] = crosstab_pct_display_formatted[col].astype(str) + '%'
                
                # TABEL PERSENTASE DENGAN WARNA MAROON
                st.markdown(create_colored_table(
                    crosstab_pct_display_formatted.reset_index(), 
                    header_color='#800000',  # <- DIUBAH DI SINI (Maroon)
                    even_color='#efebe9', 
                    odd_color='#ffffff',
                    total_color='#5a0000'  # <- DIUBAH DI SINI (Maroon lebih gelap)
                ), unsafe_allow_html=True)
                
            else:
                st.info("📊 Tidak ada data tabel untuk ditampilkan")

            st.markdown("---")

            # Statistik Tambahan
            st.markdown('<div class="section-title">🧮 Statistik Tambahan</div>', unsafe_allow_html=True)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                if 'makna_ngopi' in df_filtered.columns:
                    total_data = len(df_filtered)
                    total_makna = len(df_filtered['makna_ngopi'].dropna())
                    unique_makna = df_filtered['makna_ngopi'].nunique()
                    st.metric("Total Responden", total_data)
                    st.metric("Data Makna Valid", total_makna)
                    st.metric("Jenis Makna", unique_makna)
            
            with col_stat2:
                if 'makna_ngopi' in df_filtered.columns and 'gelas_num' in df_filtered.columns:
                    avg_gelas = df_filtered['gelas_num'].mean()
                    max_gelas = df_filtered['gelas_num'].max()
                    makna_group = safe_groupby(df_filtered, 'makna_ngopi', 'gelas_num', 'mean')
                    makna_max_konsumsi = makna_group.idxmax() if len(makna_group) > 0 else "Tidak ada data"
                    st.metric("Rata-rata Konsumsi", f"{avg_gelas:.1f} gelas/hari")
                    st.metric("Konsumsi Maksimum", f"{max_gelas:.1f} gelas/hari")
                    st.metric("Makna Konsumsi Tertinggi", f"{makna_max_konsumsi}")
            
            with col_stat3:
                if 'makna_ngopi' in df_filtered.columns and 'pengeluaran_num' in df_filtered.columns:
                    avg_pengeluaran = df_filtered['pengeluaran_num'].mean()
                    max_pengeluaran = df_filtered['pengeluaran_num'].max()
                    makna_group = safe_groupby(df_filtered, 'makna_ngopi', 'pengeluaran_num', 'mean')
                    makna_max_pengeluaran = makna_group.idxmax() if len(makna_group) > 0 else "Tidak ada data"
                    st.metric("Rata-rata Pengeluaran", f"Rp {avg_pengeluaran:,.0f}/mg")
                    st.metric("Pengeluaran Maksimum", f"Rp {max_pengeluaran:,.0f}/mg")
                    st.metric("Makna Pengeluaran Tertinggi", f"{makna_max_pengeluaran}")

# ==================== TAB 5: ANALISIS REGRESI ====================
with tab5:
    st.markdown('<div class="section-title">📈 ANALISIS 4: PENGARUH EFEK KOPI TERHADAP FOKUS BELAJAR</div>', unsafe_allow_html=True)
    
    if len(df_filtered) == 0:
        st.info("📊 Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter untuk melihat analisis.")
    else:
        if 'efek_kopi_num' in df_filtered.columns and 'fokus_num' in df_filtered.columns:
            # Dapatkan data yang valid untuk analisis regresi
            valid_data = df_filtered[['efek_kopi_num', 'fokus_num']].dropna()
            
            if len(valid_data) > 1:
                model, y_pred, r2, rmse = calculate_regression(df_filtered)
                
                if model is not None:
                    X = valid_data[['efek_kopi_num']].values
                    y = valid_data['fokus_num'].values
                    
                    # Pastikan y_pred sesuai dengan data yang valid
                    if len(y_pred) != len(y):
                        y_pred = model.predict(X)
                    
                    intercept = model.intercept_
                    slope = model.coef_[0]
                    
                    n = len(y)
                    residuals = y - y_pred
                    std_residuals = np.std(residuals, ddof=2)
                    se_slope = std_residuals / np.sqrt(np.sum((X.flatten() - X.mean())**2))
                    t_stat = slope / se_slope
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    corr = np.corrcoef(X.flatten(), y)[0, 1]
                    mae = np.mean(np.abs(y - y_pred))
                    
                    # INFO REGRESI DASAR (selalu tampil)
                    st.markdown('<div class="subsection-title">🧮 Analisis Regresi Linear</div>', unsafe_allow_html=True)
                    
                    col_basic1, col_basic2 = st.columns(2)
                    
                    with col_basic1:
                        st.markdown(f"""
                            **Persamaan Regresi:**
                            
                            `Fokus = {intercept:.2f} + {slope:.2f} × Efek Kopi`
                            
                            Setiap peningkatan 1 tingkat efek kopi meningkatkan fokus sebesar **{slope:.2f}%**
                        """)
                    
                    with col_basic2:
                        st.markdown(f"""
                            **Kualitas Model:**
                            - R² = **{r2:.4f}** ({r2*100:.1f}%)
                            - RMSE = **{rmse:.2f}%**
                            - Korelasi = **{corr:.4f}**
                            - {'✅ Signifikan' if p_value < 0.05 else '❌ Tidak Signifikan'}
                        """)

                    # STATISTIK DETAIL REGRESI (hanya jika show_statistics = True)
                    if show_statistics:
                        st.markdown("---")
                        st.markdown('<div class="section-title">📋 Statistik Detail Regresi</div>', unsafe_allow_html=True)
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Standard Error", f"{std_residuals:.2f}")
                            st.metric("Mean Absolute Error", f"{mae:.2f}")
                            st.metric("F-statistic", f"{t_stat**2:.3f}")
                        
                        with col_stat2:
                            st.metric("t-statistic", f"{t_stat:.3f}")
                            st.metric("p-value", f"{p_value:.6f}")
                            st.metric("Degrees of Freedom", f"{n-2}")
                        
                        with col_stat3:
                            st.metric("Condition Number", "Good" if len(valid_data) > 10 else "Fair")
                            st.metric("Residual Mean", f"{residuals.mean():.2f}")
                            st.metric("Jumlah Sample", f"{n}")

                    st.markdown("---")

                    # VISUALISASI REGRESI UTAMA (selalu tampil)
                    st.markdown('<div class="section-title">📊 Visualisasi Model Regresi</div>', unsafe_allow_html=True)

                    col_vis1, col_vis2 = st.columns(2)

                    with col_vis1:
                        st.markdown('<div class="subsection-title">Scatter Plot</div>', unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                        
                        # Scatter plot untuk data aktual dengan warna custom
                        ax.scatter(X, y, alpha=0.6, s=80, color='#B87333', edgecolors='#3e2723', linewidth=1)  # <- DIUBAH DI SINI (Copper)
                        
                        # Garis regresi dengan warna custom
                        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                        y_line = model.predict(x_line)
                        ax.plot(x_line, y_line, '#800000', linewidth=3, label=f'y = {intercept:.2f} + {slope:.2f}x')  # <- DIUBAH DI SINI (Maroon)
                        
                        ax.set_xlabel('Efek Kopi', fontsize=14, fontweight='600', color='#3e2723')
                        ax.set_ylabel('Fokus (%)', fontsize=14, fontweight='600', color='#3e2723')
                        
                        ax.set_xticks([-1, 0, 1, 2, 3])
                        ax.set_xticklabels(['Cemas', 'Biasa', 'Tahan', 'Semangat', 'Fokus'], fontsize=13, color='#3e2723')
                        
                        # Set batas yang sesuai
                        ax.set_ylim(bottom=max(0, y.min()-10), top=min(100, y.max()+10))
                        ax.set_xlim(-1.2, 3.2)
                        
                        ax.set_yticks(range(0, 101, 20))
                        ax.set_yticklabels(range(0, 101, 20), fontsize=13, color='#3e2723')
                        
                        ax.grid(alpha=0.2, color='#6d4c41')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#6d4c41')
                        ax.spines['bottom'].set_color('#6d4c41')
                        
                        ax.legend(fontsize=13, frameon=True, fancybox=True, shadow=True, loc='lower left')
                        
                        ax.text(0.05, 0.95, f'R² = {r2:.4f}\np = {p_value:.4f}', transform=ax.transAxes, fontsize=13,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.9, edgecolor='#6d4c41'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)

                    with col_vis2:
                        st.markdown('<div class="subsection-title">Residual Plot</div>', unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                        
                        # Residual plot dengan warna custom
                        ax.scatter(y_pred, residuals, alpha=0.6, s=80, color='#3A66A7', edgecolors='#3e2723', linewidth=1)  # <- DIUBAH DI SINI (Galaxy Blue)
                        ax.axhline(y=0, color='#800000', linestyle='--', linewidth=2.5)  # <- DIUBAH DI SINI (Maroon)
                        
                        ax.set_xlabel('Nilai Prediksi', fontsize=14, fontweight='600', color='#3e2723')
                        ax.set_ylabel('Residuals', fontsize=14, fontweight='600', color='#3e2723')
                        
                        ax.tick_params(axis='both', which='major', labelsize=13)
                        
                        ax.grid(alpha=0.2, color='#6d4c41')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#6d4c41')
                        ax.spines['bottom'].set_color('#6d4c41')
                        
                        plt.tight_layout()
                        st.pyplot(fig)

                    # DIAGNOSTIC PLOTS (hanya jika show_detailed_charts = True)
                    if show_detailed_charts:
                        st.markdown("---")
                        st.markdown('<div class="section-title">🔍 Diagnostik Model Lanjutan</div>', unsafe_allow_html=True)

                        col_vis3, col_vis4 = st.columns(2)

                        with col_vis3:
                            st.markdown('<div class="subsection-title">Q-Q Plot Residuals</div>', unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                            
                            # Q-Q plot dengan warna custom
                            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=ax)
                            ax.get_lines()[0].set_color('#C65A1E')  # <- DIUBAH DI SINI (Burnt Orange)
                            ax.get_lines()[0].set_marker('o')
                            ax.get_lines()[0].set_markersize(5)
                            ax.get_lines()[0].set_alpha(0.7)
                            ax.get_lines()[1].set_color("#000B80")  # <- DIUBAH DI SINI (Maroon)
                            ax.get_lines()[1].set_linewidth(2.5)
                            
                            ax.set_title('Q-Q Plot Residuals', fontsize=14, fontweight='600', color='#3e2723')
                            ax.set_xlabel('Theoretical Quantiles', fontsize=14, fontweight='600', color='#3e2723')
                            ax.set_ylabel('Sample Quantiles', fontsize=14, fontweight='600', color='#3e2723')
                            ax.tick_params(axis='both', which='major', labelsize=13)
                            
                            ax.grid(alpha=0.2, color='#6d4c41')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_color('#6d4c41')
                            ax.spines['bottom'].set_color('#6d4c41')
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                        with col_vis4:
                            st.markdown('<div class="subsection-title">Distribusi Residuals</div>', unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                            
                            # Histogram residuals dengan warna custom
                            ax.hist(residuals, bins=20, color='#708238', edgecolor='#3e2723', alpha=0.8, linewidth=1.5)  # <- DIUBAH DI SINI (Olive Green)
                            ax.axvline(x=0, color='#800000', linestyle='--', linewidth=2.5, label='Mean = 0')  # <- DIUBAH DI SINI (Maroon)
                            
                            ax.set_xlabel('Residuals', fontsize=14, fontweight='600', color='#3e2723')
                            ax.set_ylabel('Frequency', fontsize=14, fontweight='600', color='#3e2723')
                            ax.legend(fontsize=13, frameon=True, fancybox=True, shadow=True)
                            
                            ax.tick_params(axis='both', which='major', labelsize=13)
                            
                            ax.grid(alpha=0.2, axis='y', color='#6d4c41')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_color('#6d4c41')
                            ax.spines['bottom'].set_color('#6d4c41')
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                else:
                    st.info("📊 Tidak ada data yang cukup untuk analisis regresi")
            else:
                st.info("📊 Tidak cukup data valid untuk analisis regresi (minimal 2 data point diperlukan)")
        else:
            st.info("📊 Kolom yang diperlukan untuk analisis regresi tidak tersedia")

# ==================== TAB 6: DATA QUALITY DASHBOARD ====================
with tab6:
    st.markdown('<div class="section-title">🔍 EVALUASI KUALITAS DAN VALIDITAS DATA</div>', unsafe_allow_html=True)
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown('<div class="subsection-title">📊 Kelengkapan Data</div>', unsafe_allow_html=True)
        
        # Hitung completeness untuk setiap kolom
        completeness_data = {}
        for col in df_filtered.columns:
            total = len(df_filtered)
            non_null = df_filtered[col].count()
            completeness = (non_null / total) * 100 if total > 0 else 0
            completeness_data[col] = completeness
        
        completeness_df = pd.DataFrame({
            'Kolom': list(completeness_data.keys()),
            'Completeness (%)': list(completeness_data.values())
        }).sort_values('Completeness (%)', ascending=True)
        
        if len(completeness_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            
            # WARNA SERAGAM - PLUM
            bars = ax.barh(completeness_df['Kolom'], completeness_df['Completeness (%)'],
                          color='#7D387D', alpha=0.8, edgecolor='#3e2723', linewidth=1.2)
            
            ax.set_xlabel('Persentase (%)', fontsize=11, fontweight='600', color='#3e2723')
            ax.set_title('Tingkat Kelengkapan Kolom', fontsize=12, fontweight='600', color='#3e2723')
            
            for bar, v in zip(bars, completeness_df['Completeness (%)']):
                ax.text(v + 1, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', 
                       va='center', fontweight='600', fontsize=9, color='#3e2723')
            
            ax.grid(axis='x', alpha=0.2, color='#6d4c41')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#6d4c41')
            ax.spines['bottom'].set_color('#6d4c41')
            st.pyplot(fig)
        else:
            st.info("📊 Tidak ada data completeness untuk ditampilkan")
    
    with col_q2:
        st.markdown('<div class="subsection-title">📈 Pengecekan Distribusi Data</div>', unsafe_allow_html=True)
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
            
            if selected_col:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
                
                # Histogram dengan warna seragam - PLUM
                ax1.hist(df_filtered[selected_col].dropna(), bins=15, 
                        color='#7D387D', alpha=0.7, edgecolor='#3e2723', linewidth=1.2)
                ax1.set_title(f'Distribusi {selected_col}', fontweight='600', color='#3e2723')
                ax1.set_xlabel(selected_col, color='#3e2723')
                ax1.set_ylabel('Frekuensi', color='#3e2723')
                
                # Box plot dengan warna seragam - PLUM
                box_plot = ax2.boxplot(df_filtered[selected_col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor='#7D387D', alpha=0.8, linewidth=1.2),
                           medianprops=dict(color='#3e2723', linewidth=2),
                           whiskerprops=dict(color='#3e2723', linewidth=1.2),
                           capprops=dict(color='#3e2723', linewidth=1.2),
                           flierprops=dict(marker='o', markersize=5, markerfacecolor='#7D387D', 
                                         markeredgecolor='#3e2723', alpha=0.7))
                ax2.set_title(f'Box Plot {selected_col}', fontweight='600', color='#3e2723')
                ax2.set_ylabel(selected_col, color='#3e2723')
                
                for ax in [ax1, ax2]:
                    ax.grid(alpha=0.2, axis='y', color='#6d4c41')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#6d4c41')
                    ax.spines['bottom'].set_color('#6d4c41')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistik - TETAP STANDARD STREAMLIT
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Mean", f"{df_filtered[selected_col].mean():.2f}")
                with col_stats2:
                    st.metric("Median", f"{df_filtered[selected_col].median():.2f}")
                with col_stats3:
                    st.metric("Std Dev", f"{df_filtered[selected_col].std():.2f}")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; font-size: 0.9em; margin-top: 2rem; padding: 2rem; background: #24201C; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); border: 1px solid #2E2A25;">
        <p style="margin-bottom: 0.8rem; font-weight: 600; color: #E0C9A6; font-size: 1em; letter-spacing: 0.5px;">
            Dashboard Survei Ngopi Mahasiswa  |  Data: {len(df)} Responden  |  Filter: {len(df_filtered)} Responden
        </p>
        <p style="margin: 0; font-weight: 400; color: #E0C9A6;">
            Dibuat dengan <span style="color: #8A2BE2; text-shadow: 0 0 8px rgba(138, 43, 226, 0.5);">Streamlit</span>  |  Advanced Analytics  |  Real-time Reporting  |  Data Quality Monitoring
        </p>
        <p style="margin-top: 1rem; font-size: 0.8em; color: #E6A65D; font-style: italic; letter-spacing: 0.3px;">
            Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
""", unsafe_allow_html=True)

# ==================== AUTO-REFRESH FEATURE ====================

if auto_refresh:
    st.markdown("---")
    refresh_placeholder = st.empty()
    with refresh_placeholder:
        st.info("🔄 Auto-refresh aktif - Data akan diperbarui setiap 30 detik...")
    
    time.sleep(30)
    st.rerun()
