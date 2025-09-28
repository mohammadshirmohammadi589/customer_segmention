import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# تنظیمات CSS سفارشی برای ظاهر حرفه‌ای و خوانا
st.markdown("""
    <style>
    /* استایل کلی */
    body {
        font-family: 'Vazir', sans-serif;
        background-color: #F5F7FA;
        color: #1A2E4A;
    }
    .stApp {
        background-color: #F5F7FA;
        color: #1A2E4A;
    }
    /* استایل سایدبار */
    .css-1d391kg {
        background-color: #1A2E4A;
        color: #FFFFFF;
        font-family: 'Vazir', sans-serif;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #FFFFFF;
    }
    /* استایل عنوان اصلی */
    .stTitle {
        color: #1A2E4A;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: right;
    }
    /* استایل تب‌ها */
    .stTabs [data-baseweb="tab"] {
        background-color: #E8ECEF;
        color: #1A2E4A;
        font-weight: 500;
        border-radius: 8px;
        margin: 5px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1A2E4A;
        color: #FFFFFF;
    }
    /* استایل دکمه‌ها */
    .stButton>button {
        background-color: #00A8B5;
        color: #FFFFFF;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #007B85;
        color: #FFFFFF;
    }
    /* استایل متریک‌ها */
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1A2E4A !important;
    }
    .stMetric label {
        color: #1A2E4A !important;
        font-weight: 600;
    }
    .stMetric .css-1x0d4go {
        color: #1A2E4A !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    /* استایل اکسپندر */
    .stExpander {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1A2E4A;
    }
    /* استایل نوشته‌های عمومی */
    .stMarkdown, .stText, .stDataFrame {
        color: #1A2E4A;
    }
    </style>
""", unsafe_allow_html=True)

# تنظیمات صفحه
st.set_page_config(
    page_title="🎯 داشبورد تحلیل RFM و خوشه‌بندی مشتریان",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# عنوان اصلی
st.title("🎯 داشبورد هوشمند تحلیل RFM و خوشه‌بندی مشتریان")
st.markdown("""
**امکانات پیشرفته:**
- ✨ تحلیل RFM خودکار با پارامترهای قابل تنظیم
- 📊 خوشه‌بندی هوشمند با K-Means
- 🎯 ویژوال‌سازی حرفه‌ای و تعاملی
- 💾 دانلود نتایج و گزارشات
- ⚡ محاسبات بلادرنگ
""")
st.markdown("---")

# ================== بخش ۱: بارگذاری داده ==================
st.sidebar.header("📤 بارگذاری داده")

def create_sample_data():
    """ایجاد داده‌های نمونه برای تست"""
    np.random.seed(42)
    n_customers = 1000
    data = {
        'CustomerID': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
        'Recency': np.random.randint(1, 365, n_customers),
        'Frequency': np.random.randint(1, 50, n_customers),
        'Monetary': np.random.randint(1000, 50000, n_customers),
        'LastPurchaseDate': pd.date_range('2023-01-01', periods=n_customers, freq='D')
    }
    return pd.DataFrame(data)

uploaded_file = st.sidebar.file_uploader(
    "فایل داده مشتریان را آپلود کنید (CSV یا Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="ستون‌های لازم: CustomerID, Recency, Frequency, Monetary"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ فایل با موفقیت آپلود شد!")
    except Exception as e:
        st.sidebar.error(f"❌ خطا در بارگذاری فایل: {e}")
        df = create_sample_data()
        st.sidebar.info("ℹ️ از داده نمونه استفاده می‌شود")
else:
    df = create_sample_data()
    st.sidebar.info("ℹ️ از داده نمونه برای نمایش استفاده می‌شود")

# نمایش داده خام
with st.expander("📊 نمایش داده‌های خام"):
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"تعداد مشتریان: {len(df):,}")

# بررسی ستون‌ها
st.sidebar.header("🔍 بررسی ستون‌ها")
st.sidebar.write("ستون‌های موجود:", df.columns.tolist())

if 'Recency' not in df.columns:
    st.sidebar.warning("❌ ستون Recency پیدا نشد! در حال ایجاد داده نمونه...")
    def create_correct_sample_data():
        np.random.seed(42)
        n_customers = 500
        data = {
            'CustomerID': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
            'Recency': np.random.randint(1, 365, n_customers),
            'Frequency': np.random.randint(1, 50, n_customers),
            'Monetary': np.random.randint(1000, 50000, n_customers)
        }
        return pd.DataFrame(data)
    df = create_correct_sample_data()
    st.sidebar.success("✅ داده نمونه با ستون‌های صحیح ایجاد شد")

st.sidebar.success("✅ ستون Recency وجود دارد")

# ================== بخش ۲: تنظیمات RFM ==================
st.sidebar.header("⚙️ تنظیمات تحلیل RFM")
st.sidebar.subheader("📊 پارامترهای چارک‌ها")

r_q1 = st.sidebar.slider("چارک اول Recency", 0, 365, 17, 1)
r_q2 = st.sidebar.slider("چارک دوم Recency", 0, 365, 49, 1)
r_q3 = st.sidebar.slider("چارک سوم Recency", 0, 365, 134, 1)

f_q1 = st.sidebar.slider("چارک اول Frequency", 0, 100, 17, 1)
f_q2 = st.sidebar.slider("چارک دوم Frequency", 0, 200, 40, 1)
f_q3 = st.sidebar.slider("چارک سوم Frequency", 0, 500, 97, 1)

m_q1 = st.sidebar.slider("چارک اول Monetary", 0, 5000, 296, 10)
m_q2 = st.sidebar.slider("چارک دوم Monetary", 0, 10000, 642, 10)
m_q3 = st.sidebar.slider("چارک سوم Monetary", 0, 20000, 1554, 10)

# ================== بخش ۳: محاسبه امتیازات RFM ==================
def calculate_rfm_scores(data, r_params, f_params, m_params):
    df = data.copy()
    def r_score(x):
        if pd.isna(x): return 1
        if x <= r_params[0]: return 4
        elif x <= r_params[1]: return 3
        elif x <= r_params[2]: return 2
        else: return 1
    def fm_score(x, params):
        if pd.isna(x): return 1
        if x <= params[0]: return 1
        elif x <= params[1]: return 2
        elif x <= params[2]: return 3
        else: return 4
    df['R_Score'] = df['Recency'].apply(r_score)
    df['F_Score'] = df['Frequency'].apply(fm_score, args=(f_params,))
    df['M_Score'] = df['Monetary'].apply(fm_score, args=(m_params,))
    df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    return df

r_params = [r_q1, r_q2, r_q3]
f_params = [f_q1, f_q2, f_q3]
m_params = [m_q1, m_q2, m_q3]

rfm_df = calculate_rfm_scores(df, r_params, f_params, m_params)

# ================== بخش ۴: خوشه‌بندی ==================
st.sidebar.header("🎯 تنظیمات خوشه‌بندی")

n_clusters = st.sidebar.slider(
    "تعداد خوشه‌ها",
    min_value=2, max_value=8, value=4, step=1
)

max_iter = st.sidebar.slider(
    "تعداد تکرار",
    min_value=100, max_value=1000, value=300, step=50
)

random_state = st.sidebar.slider(
    "مقدار تصادفی",
    min_value=0, max_value=100, value=42, step=1
)

def perform_clustering(data, n_clusters, max_iter, random_state):
    features = ['Recency', 'Frequency', 'Monetary']
    # بررسی وجود ستون‌ها
    if not all(col in data.columns for col in features):
        st.error("❌ ستون‌های لازم برای خوشه‌بندی وجود ندارند!")
        return data, None, None
    # حذف مقادیر NaN
    data = data.dropna(subset=features)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10
    )
    clusters = kmeans.fit_predict(scaled_data)
    data['Cluster'] = clusters
    silhouette_avg = silhouette_score(scaled_data, clusters) if len(data) > 1 else None
    return data, silhouette_avg, kmeans.cluster_centers_

clustered_df, silhouette_avg, centers = perform_clustering(
    rfm_df, n_clusters, max_iter, random_state
)

# ================== بخش ۵: نمایش نتایج ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 خلاصه کلی",
    "📈 نمودارها",
    "🎯 تحلیل خوشه‌ها",
    "🏆 رنک‌بندی",
    "💾 خروجی"
])

with tab1:
    st.header("📊 خلاصه کلی تحلیل")
    
    # دیباگ: نمایش مقادیر برای بررسی
    st.write("**دیباگ مقادیر متریک‌ها:**")
    st.write(f"تعداد مشتریان: {len(clustered_df)}")
    st.write(f"تعداد خوشه‌ها: {n_clusters}")
    st.write(f"میانگین Recency: {clustered_df['Recency'].mean() if not clustered_df.empty else 'نامعتبر'}")
    st.write(f"میانگین Frequency: {clustered_df['Frequency'].mean() if not clustered_df.empty else 'نامعتبر'}")
    st.write(f"میانگین Monetary: {clustered_df['Monetary'].mean() if not clustered_df.empty else 'نامعتبر'}")
    st.write(f"کل Monetary: {clustered_df['Monetary'].sum() if not clustered_df.empty else 'نامعتبر'}")
    st.write(f"Silhouette Score: {silhouette_avg if silhouette_avg is not None else 'نامعتبر'}")
    st.write(f"توزیع خوشه‌ها: {clustered_df['Cluster'].value_counts().to_dict() if not clustered_df.empty else 'نامعتبر'}")
    
    # نمایش متریک‌ها
    if clustered_df.empty:
        st.error("❌ داده‌ها خالی هستند! لطفاً داده‌های ورودی را بررسی کنید.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="تعداد مشتریان", value=str(len(clustered_df)) if len(clustered_df) > 0 else "0")
            st.metric(label="تعداد خوشه‌ها", value=str(n_clusters))
        
        with col2:
            avg_r = clustered_df['Recency'].mean()
            avg_f = clustered_df['Frequency'].mean()
            st.metric(label="میانگین Recency", value=f"{avg_r:.1f} روز" if not pd.isna(avg_r) else "نامعتبر")
            st.metric(label="میانگین Frequency", value=f"{avg_f:.1f}" if not pd.isna(avg_f) else "نامعتبر")
        
        with col3:
            avg_m = clustered_df['Monetary'].mean()
            total_m = clustered_df['Monetary'].sum()
            st.metric(label="میانگین Monetary", value=f"{avg_m:,.0f}" if not pd.isna(avg_m) else "نامعتبر")
            st.metric(label="کل Monetary", value=f"{total_m:,.0f}" if not pd.isna(total_m) else "نامعتبر")
        
        with col4:
            st.metric(label="Silhouette Score", value=f"{silhouette_avg:.3f}" if silhouette_avg is not None else "نامعتبر")
            cluster_counts = clustered_df['Cluster'].value_counts()
            if not cluster_counts.empty:
                st.metric(label="بزرگترین خوشه", value=f"{cluster_counts.idxmax()} ({cluster_counts.max()})")
            else:
                st.metric(label="بزرگترین خوشه", value="نامعتبر")

with tab2:
    st.header("📈 نمودارهای تحلیلی")
    
    col1, col2 = st.columns(2)
    
    with col1:
        score_dist = clustered_df['RFM_Score'].value_counts().head(10)
        fig_scores = px.bar(
            x=score_dist.values, y=score_dist.index,
            orientation='h', title='۱۰ امتیاز برتر RFM',
            labels={'x': 'تعداد مشتریان', 'y': 'امتیاز RFM'},
            color_discrete_sequence=['#00A8B5']
        )
        fig_scores.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family="Vazir", size=12, color="#1A2E4A"),
            title_font=dict(size=18, family="Vazir", color="#1A2E4A")
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        cluster_dist = clustered_df['Cluster'].value_counts()
        fig_clusters = px.pie(
            values=cluster_dist.values, names=cluster_dist.index,
            title='توزیع مشتریان بین خوشه‌ها',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_clusters.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family="Vazir", size=12, color="#1A2E4A"),
            title_font=dict(size=18, family="Vazir", color="#1A2E4A")
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    
    st.subheader("📊 نمودار پراکندگی")
    col_x, col_y, col_color = st.columns(3)
    
    with col_x:
        x_axis = st.selectbox("محور X", ['Recency', 'Frequency', 'Monetary'], key='x_axis')
    with col_y:
        y_axis = st.selectbox("محور Y", ['Recency', 'Frequency', 'Monetary'], key='y_axis')
    with col_color:
        color_by = st.selectbox("رنگ بر اساس", ['Cluster', 'R_Score', 'F_Score', 'M_Score'], key='color_by')
    
    fig_scatter = px.scatter(
        clustered_df, x=x_axis, y=y_axis, color=color_by,
        title=f'{y_axis} در مقابل {x_axis}',
        hover_data=['RFM_Score', 'CustomerID'],
        size='Monetary' if 'Monetary' not in [x_axis, y_axis] else None,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_scatter.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(family="Vazir", size=12, color="#1A2E4A"),
        title_font=dict(size=18, family="Vazir", color="#1A2E4A")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("🎯 تحلیل خوشه‌ها")
    
    cluster_stats = clustered_df.groupby('Cluster').agg({
        'Recency': ['mean', 'std', 'count'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std', 'sum'],
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean'
    }).round(2)
    
    st.dataframe(cluster_stats, use_container_width=True)
    
    st.subheader("📊 مقایسه معیارها بین خوشه‌ها")
    selected_metrics = st.multiselect(
        "انتخاب معیارها برای مقایسه",
        ['Recency', 'Frequency', 'Monetary'],
        default=['Recency', 'Frequency', 'Monetary']
    )
    
    if selected_metrics:
        fig_comparison = go.Figure()
        for metric in selected_metrics:
            fig_comparison.add_trace(go.Box(
                y=clustered_df[metric], x=clustered_df['Cluster'],
                name=metric, boxpoints='outliers'
            ))
        
        fig_comparison.update_layout(
            title='مقایسه معیارها بین خوشه‌ها',
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family="Vazir", size=12, color="#1A2E4A"),
            title_font=dict(size=18, family="Vazir", color="#1A2E4A")
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

with tab4:
    st.header("🏆 رنک‌بندی مشتریان")
    
    rank_by = st.selectbox(
        "رنک‌بندی بر اساس:",
        ['Monetary', 'Frequency', 'Recency', 'RFM_Score'],
        index=0
    )
    
    n_top = st.slider("تعداد مشتریان نمایش داده شده", 10, 100, 20, 5)
    
    ranked_df = clustered_df.nlargest(n_top, rank_by)
    
    st.dataframe(
        ranked_df[[
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Cluster'
        ]].style.background_gradient(
            subset=['Monetary', 'Frequency'],
            cmap='YlOrBr'
        ),
        use_container_width=True,
        height=600
    )

with tab5:
    st.header("💾 خروجی و دانلود")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 داده‌های خوشه‌بندی شده")
        st.dataframe(clustered_df.head(), use_container_width=True)
        
        csv = clustered_df.to_csv(index=False)
        st.download_button(
            label="📥 دانلود داده‌ها (CSV)",
            data=csv,
            file_name=f"rfm_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📈 گزارش تحلیل")
        report = f"""
# 📊 گزارش تحلیل RFM و خوشه‌بندی
تاریخ تولید: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 📈 آمار کلی
- تعداد مشتریان: {len(clustered_df):,}
- تعداد خوشه‌ها: {n_clusters}
- کیفیت خوشه‌بندی (Silhouette Score): {silhouette_avg if silhouette_avg is not None else 'نامعتبر'}

## 🎯 پارامترهای استفاده شده
- چارک‌های Recency: {r_params}
- چارک‌های Frequency: {f_params}
- چارک‌های Monetary: {m_params}
- تعداد خوشه‌ها: {n_clusters}
- تعداد تکرار: {max_iter}

## 📊 نتایج کلیدی
- میانگین Recency: {clustered_df['Recency'].mean():.1f} روز
- میانگین Frequency: {clustered_df['Frequency'].mean():.1f}
- میانگین Monetary: {clustered_df['Monetary'].mean():,.0f}
- کل ارزش Monetary: {clustered_df['Monetary'].sum():,.0f}

## 🎪 توزیع خوشه‌ها
{clustered_df['Cluster'].value_counts().to_string()}

## 🏆 مشتریان برتر
{clustered_df.nlargest(5, 'Monetary')[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']].to_string()}
"""
        st.text_area("گزارش تولید شده:", report, height=300)
        st.download_button(
            label="📄 دانلود گزارش (TXT)",
            data=report,
            file_name=f"rfm_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

# ================== بخش ۶: اطلاعات پایین صفحه ==================
st.markdown("---")
st.markdown("""
**🎯 راهنما:**
- از سایدبار برای تغییر پارامترها استفاده کنید
- نتایج به صورت زنده به‌روزرسانی می‌شوند
- از تب‌های مختلف برای دیدگاه‌های مختلف استفاده کنید
- می‌توانید نتایج را دانلود کنید

**📞 پشتیبانی:** برای سوالات با ما تماس بگیرید
**توسعه‌دهنده:** تیم تحلیل داده xAI
""")

st.sidebar.info("ℹ️ با تغییر هر پارامتر، نتایج به صورت خودکار به‌روزرسانی می‌شوند")
