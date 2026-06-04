import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
import requests
import warnings
import time

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Real Estate Intelligence Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0px 0px 20px rgba(0,255,170,0.2);
}

.stButton>button {
    background: linear-gradient(90deg,#00FFAA,#00C2FF);
    color: black;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* GLASSMORPHISM CARDS */
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
    margin-bottom: 20px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(17,25,40,0.95);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* KPI METRICS */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 30px rgba(0,0,0,0.2);
    transition: 0.3s;
}

/* HOVER EFFECT */
[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 40px rgba(0,255,170,0.3);
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #00FFAA, #00C2FF);
    color: black;
    border-radius: 12px;
    font-weight: bold;
    border: none;
    padding: 12px 24px;
    transition: 0.3s;
}

/* BUTTON HOVER */
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0,255,170,0.5);
}

/* HEADINGS */
h1, h2, h3 {
    color: white;
    font-weight: 700;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #00FFAA;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# GOOGLE ANALYTICS
# =====================================================

GA_ID = "G-8JJ3T5FVXN"

html(f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>

<script>
window.dataLayer = window.dataLayer || [];

function gtag(){{
dataLayer.push(arguments);
}}

gtag('js', new Date());

gtag('config', '{GA_ID}');
</script>
""", height=0)

# =====================================================
# LOGIN SYSTEM
# =====================================================

st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

# =====================================================
# LOGIN CHECK
# =====================================================

if username == "admin" and password == "admin123":

    st.sidebar.success("Login Successful")

    # =====================================================
    # TITLE
    # =====================================================

    st.title("🏙 Real Estate Intelligence Hub")

    st.markdown("""
    ## AI-Powered Property Analytics & Investment Intelligence Platform

    Analyze market trends, predict housing prices, discover investment opportunities,
    and explore advanced real estate insights through interactive AI-driven dashboards.

    ### 🚀 Key Features
    - AI House Price Prediction
    - Interactive Visual Analytics
    - Smart Investment Recommendations
    - Real Estate Market Forecasting
    - Advanced Geospatial Mapping
    - AI-Powered Insights & Reports
    """)
    st.success("Dashboard Loaded Successfully")

    # =====================================================
    # LOAD DATA
    # =====================================================

    st.sidebar.header("📂 Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/kc_house_data.csv")
        
    st.sidebar.write("### Dataset Information")
    st.sidebar.write(f"Rows: {df.shape[0]}")
    st.sidebar.write(f"Columns: {df.shape[1]}")
    
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # =====================================================
    # MACHINE LEARNING MODEL
    # =====================================================

    X = df[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =====================================================
    # SIDEBAR MENU
    # =====================================================

    with st.sidebar:

        selected = option_menu(
            menu_title="🏠 AI Real Estate",

            options=[
                "Dashboard",
                "Dataset",
                "Price Analysis",
                "Area Analysis",
                "Investment Insights",
                "Compare Houses",
                "AI Recommendations",
                "AI Explainability",
                "Forecasting",
                "Animated Charts",
                "Map Visualization",
                "Model Comparison",
                "Weather",
                "AI Chatbot",
                "Downloads"
            ],

            icons=[
                "house",
                "table",
                "graph-up",
                "geo-alt",
                "cash-stack",
                "columns-gap",
                "robot",
                "cpu",
                "bar-chart-line",
                "pie-chart",
                "map",
                "diagram-3",
                "cloud-sun",
                "chat-dots",
                "download"
            ],

            default_index=0
        )

    # =====================================================
    # FUNCTIONS
    # =====================================================

    def investment_score(row):

        score = 0

        if row['grade'] >= 8:
            score += 30

        if row['waterfront'] == 1:
            score += 30

        if row['sqft_living'] > 3000:
            score += 20

        if row['condition'] >= 4:
            score += 20

        return score

    def create_pdf():

        doc = SimpleDocTemplate("report.pdf")

        styles = getSampleStyleSheet()

        elements = []

        elements.append(
            Paragraph(
                "Real Estate Analytics Report",
                styles['Title']
            )
        )

        doc.build(elements)

    def recommend_houses(price, bedrooms):

        rec = df[
            (df['price'] <= price)
            &
            (df['bedrooms'] == bedrooms)
        ]

        return rec.head(10)

    # =====================================================
    # DASHBOARD
    # =====================================================

    if selected == "Dashboard":

        st.header("📊 AI Real Estate Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Average Price",
                f"${int(df['price'].mean()):,}"
            )

        with col2:
            st.metric(
                "Maximum Price",
                f"${int(df['price'].max()):,}"
            )

        with col3:
            st.metric(
                "Total Houses",
                len(df)
            )

        with col4:
            st.metric(
                "Average Bedrooms",
                round(df['bedrooms'].mean(), 2)
            )

        style_metric_cards()

        fig = px.scatter(
            df,
            x='sqft_living',
            y='price',
            color='grade',
            size='bathrooms',
            hover_data=['zipcode'],
            title='AI Housing Market Analysis'
        )
        
        x_axis = st.selectbox(
            "Select X-axis",
            numeric_columns
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_columns
        )
        
        fig2 = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=y_axis,
            title="Interactive AI Visualization"
        )

        st.plotly_chart(fig2)
        
        features = st.multiselect(
            "Select Feature Columns",
            numeric_columns,
            default=numeric_columns[:3]
        )
        
        if len(features) >= 2:
            fig3 = px.scatter_matrix(
                df,
                dimensions=features,
                color='price',
                title='AI Feature Correlation Matrix'
            )
            st.plotly_chart(fig3)
        
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("📊 Statistical Summary")
        st.write(df.describe())
        
        st.subheader("❓ Missing Values")
        st.write(df.isnull().sum())
        
        st.subheader("🧠 AI Insights")
        st.write(
            f"""
            Dataset contains {df.shape[0]} rows
            and {df.shape[1]} columns.
            """
        )

        st.plotly_chart(fig, use_container_width=True)

        fig3d = px.scatter_3d(
            df,
            x='sqft_living',
            y='bedrooms',
            z='price',
            color='grade'
        )

        st.plotly_chart(fig3d, use_container_width=True)

    # =====================================================
    # DATASET
    # =====================================================

    elif selected == "Dataset":

        st.header("📊 Dataset Preview")

        st.dataframe(df.head())

        st.write("Shape:", df.shape)

        st.write("Columns:")

        st.write(df.columns.tolist())

        st.write("Missing Values:")

        st.write(df.isnull().sum())

    # =====================================================
    # PRICE ANALYSIS
    # =====================================================

    elif selected == "Price Analysis":

        st.header("💰 House Price Distribution")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(df['price'], kde=True, ax=ax)

        st.pyplot(fig)

        st.subheader("Top 10 Expensive Houses")

        st.dataframe(
            df.nlargest(10, 'price')[
                [
                    'price',
                    'bedrooms',
                    'bathrooms',
                    'sqft_living',
                    'zipcode'
                ]
            ]
        )

    # =====================================================
    # AREA ANALYSIS
    # =====================================================

    elif selected == "Area Analysis":

        st.header("📍 Area-wise Analysis")

        zipcode = st.selectbox(
            "Select Zipcode",
            sorted(df['zipcode'].unique())
        )

        filtered = df[df['zipcode'] == zipcode]

        st.dataframe(filtered.head())

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            x='sqft_living',
            y='price',
            data=filtered,
            ax=ax
        )

        st.pyplot(fig)

    # =====================================================
    # INVESTMENT INSIGHTS
    # =====================================================

    elif selected == "Investment Insights":

        st.header("📈 Investment Analysis")

        df['investment_score'] = df.apply(
            investment_score,
            axis=1
        )

        st.dataframe(
            df.sort_values(
                by='investment_score',
                ascending=False
            )[[
                'price',
                'zipcode',
                'investment_score',
                'sqft_living',
                'grade',
                'condition'
            ]].head(10)
        )

    # =====================================================
    # COMPARE HOUSES
    # =====================================================

    elif selected == "Compare Houses":

        st.header("🏡 House Comparison")

        house1 = st.selectbox("House A", df.index)

        house2 = st.selectbox("House B", df.index)

        h1 = df.loc[house1]

        h2 = df.loc[house2]

        comparison = pd.DataFrame({

            "Feature": [
                "Price",
                "Bedrooms",
                "Bathrooms",
                "Sqft",
                "Floors",
                "Condition",
                "Grade"
            ],

            "House A": [
                h1['price'],
                h1['bedrooms'],
                h1['bathrooms'],
                h1['sqft_living'],
                h1['floors'],
                h1['condition'],
                h1['grade']
            ],

            "House B": [
                h2['price'],
                h2['bedrooms'],
                h2['bathrooms'],
                h2['sqft_living'],
                h2['floors'],
                h2['condition'],
                h2['grade']
            ]
        })

        st.dataframe(comparison)

    # =====================================================
    # AI RECOMMENDATIONS
    # =====================================================

    elif selected == "AI Recommendations":

        st.header("🤖 AI Recommendations")

        st.subheader("Luxury Homes")

        st.dataframe(
            df[df['price'] > 1000000].head(10)
        )

        st.subheader("Undervalued Properties")

        df['price_per_sqft'] = (
            df['price'] / df['sqft_living']
        )

        st.dataframe(
            df.sort_values('price_per_sqft').head(10)
        )

        price = st.slider(
            "Budget",
            50000,
            2000000,
            500000
        )

        bedrooms = st.slider(
            "Bedrooms",
            1,
            10,
            3
        )

        recommendations = recommend_houses(price, bedrooms)

        st.dataframe(
            recommendations[
                [
                    'price',
                    'bedrooms',
                    'bathrooms',
                    'sqft_living',
                    'zipcode'
                ]
            ]
        )

    # =====================================================
    # AI EXPLAINABILITY
    # =====================================================

    elif selected == "AI Explainability":

        st.header("🧠 AI Explainability")

        feature_importance = pd.DataFrame({

            "Feature": X.columns,

            "Importance": model.feature_importances_

        }).sort_values(
            by="Importance",
            ascending=False
        )

        st.bar_chart(
            feature_importance.set_index("Feature")
        )

    # =====================================================
    # FORECASTING
    # =====================================================

    elif selected == "Forecasting":

        st.header("📈 House Price Forecasting")

        if "date" in df.columns:

            df['year'] = pd.to_datetime(df['date']).dt.year

            yearly_prices = (
                df.groupby('year')['price']
                .mean()
                .reset_index()
            )

            fig = px.line(
                yearly_prices,
                x='year',
                y='price',
                title='Future Market Trend'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # ANIMATED CHARTS
    # =====================================================

    elif selected == "Animated Charts":

        st.header("📊 Interactive Animated Charts")

        fig = px.histogram(
            df,
            x='price',
            nbins=50
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig2 = px.scatter(
            df,
            x='sqft_living',
            y='price',
            color='bedrooms'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =====================================================
    # MAP VISUALIZATION
    # =====================================================

    elif selected == "Map Visualization":

        st.header("🗺 House Location Map")

        m = folium.Map(
            location=[
                df['lat'].mean(),
                df['long'].mean()
            ],
            zoom_start=9
        )

        marker_cluster = MarkerCluster().add_to(m)

        for i in range(min(300, len(df))):

            folium.CircleMarker(
                location=[
                    df.iloc[i]['lat'],
                    df.iloc[i]['long']
                ],

                radius=5,

                popup=f"""
                Price: ${df.iloc[i]['price']}
                <br>
                Bedrooms: {df.iloc[i]['bedrooms']}
                """,

                color='blue',

                fill=True,

                fill_color='blue'

            ).add_to(marker_cluster)

        st_folium(
            m,
            width=1200,
            height=600
        )

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    elif selected == "Model Comparison":

        st.header("🤖 Model Performance")

        models = {

            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(),

            "XGBoost": XGBRegressor()
        }

        results = {}

        for model_name, model_obj in models.items():

            model_obj.fit(X_train, y_train)

            score = model_obj.score(X_test, y_test)

            results[model_name] = score

        st.dataframe(
            pd.DataFrame.from_dict(
                results,
                orient='index',
                columns=['R² Score']
            )
        )

    # =====================================================
    # WEATHER
    # =====================================================

    elif selected == "Weather":

        st.header("🌦 Seattle Weather")

        API_KEY = "YOUR_API_KEY"

        url = f"https://api.openweathermap.org/data/2.5/weather?q=Seattle&appid={API_KEY}&units=metric"

        try:

            response = requests.get(url)

            if response.status_code == 200:

                data = response.json()

                weather = data['weather'][0]['description']

                temp = data['main']['temp']

                humidity = data['main']['humidity']

                st.write(f"Weather: {weather}")

                st.write(f"Temperature: {temp} °C")

                st.write(f"Humidity: {humidity}%")

            else:

                st.write("Could not fetch weather data.")

        except Exception as e:

            st.error(f"Weather API Error: {e}")

    # =====================================================
    # AI CHATBOT
    # =====================================================

    elif selected == "AI Chatbot":

        st.header("💬 Real Estate AI Assistant")

        user_question = st.text_input("Ask AI")

        if user_question:

            if "investment" in user_question.lower():

                st.write(
                    "AI: Seattle suburbs show strong long-term growth potential."
                )

            elif "luxury" in user_question.lower():

                st.write(
                    "AI: Waterfront properties are premium investments."
                )

            else:

                st.write(
                    "AI: Market conditions currently favor medium-term investment."
                )

    # =====================================================
    # DOWNLOADS
    # =====================================================

    elif selected == "Downloads":

        st.header("📥 Download Reports")

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download CSV",
            csv,
            "kc_house_data.csv",
            "text/csv"
        )

        create_pdf()

        with open("report.pdf", "rb") as file:

            st.download_button(
                "Download PDF",
                file,
                file_name="analytics_report.pdf"
            )

    st.markdown("---")

    st.markdown(
        "### 🚀 Developed using AI + Machine Learning + Streamlit"
    )
    
else:
    st.warning("Please login to continue")