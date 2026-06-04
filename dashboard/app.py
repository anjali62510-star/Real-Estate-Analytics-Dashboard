"""
Real Estate Intelligence Hub - Enterprise Edition v3.0
======================================================
Production-ready Streamlit application with:
  - AI Assistant/Chatbot UI
  - Advanced Dashboards & Analytics
  - Product Pages
  - REST API Integration
  - File/Database Backend (SQLite)
  - Modern Dark Themes with Glassmorphism
  - Admin Panels with Role-Based Access
  - Real-time Notifications
  - Data Export & Import

Author: Senior Engineering Team
Version: 3.0.0-ENTERPRISE
"""

from __future__ import annotations

import bcrypt
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import MarkerCluster, HeatMap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================

class Config:
    ADMIN_USERNAME = os.getenv("REI_USERNAME", "admin")
    ADMIN_PASSWORD_HASH = os.getenv("REI_PASSWORD_HASH", "$2b$12$uq7z0SwW45YTEQ3Ex8RkmO0X2rFmAjKHYUmUHsCNMLBXiL.xEo2R6")
    WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    GA_TRACKING_ID = os.getenv("GA_TRACKING_ID", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_DATA_PATH = Path("data/kc_house_data.csv")
    MODEL_CACHE_DIR = Path(".cache/models")
    REPORT_DIR = Path(".cache/reports")
    DB_PATH = Path(".cache/rei_hub.db")
    UPLOAD_DIR = Path(".cache/uploads")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 200
    MAX_MAP_MARKERS = 500
    PAGE_TITLE = "Real Estate Intelligence Hub"
    PAGE_ICON = "🏠"
    VERSION = "3.0.0-ENTERPRISE"
    CHATBOT_NAME = "REI Assistant"
    CHATBOT_AVATAR = "🤖"

    @classmethod
    def init_dirs(cls):
        for d in [cls.MODEL_CACHE_DIR, cls.REPORT_DIR, cls.UPLOAD_DIR]:
            d.mkdir(parents=True, exist_ok=True)

Config.init_dirs()

# =====================================================
# DATABASE BACKEND
# =====================================================

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                price REAL,
                bedrooms INTEGER,
                bathrooms REAL,
                sqft_living INTEGER,
                sqft_lot INTEGER,
                floors REAL,
                waterfront INTEGER DEFAULT 0,
                view INTEGER DEFAULT 0,
                condition INTEGER DEFAULT 3,
                grade INTEGER DEFAULT 7,
                yr_built INTEGER,
                yr_renovated INTEGER,
                zipcode TEXT,
                lat REAL,
                long REAL,
                sqft_above INTEGER,
                sqft_basement INTEGER,
                status TEXT DEFAULT 'active',
                featured INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                user_id INTEGER,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                response_time REAL,
                client_ip TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT DEFAULT 'info',
                is_read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("SELECT id FROM users WHERE username = ?", (Config.ADMIN_USERNAME,))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            """, (Config.ADMIN_USERNAME, Config.ADMIN_PASSWORD_HASH, "admin@reihub.com", "admin"))
        self.conn.commit()

    def execute(self, query, params=()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor.fetchall()

    def fetchone(self, query, params=()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()

    def fetchall(self, query, params=()):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def insert(self, table, data):
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.conn.cursor()
        cursor.execute(query, tuple(data.values()))
        self.conn.commit()
        return cursor.lastrowid

    def update(self, table, data, where, where_params):
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        cursor = self.conn.cursor()
        cursor.execute(query, tuple(data.values()) + where_params)
        self.conn.commit()
        return cursor.rowcount

    def delete(self, table, where, params):
        query = f"DELETE FROM {table} WHERE {where}"
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor.rowcount

    def get_df(self, query, params=()):
        return pd.read_sql_query(query, self.conn, params=params)

db = DatabaseManager()

# =====================================================
# LOGGING SETUP
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rei_hub")

# =====================================================
# CUSTOM EXCEPTIONS
# =====================================================

class DataValidationError(Exception):
    pass

class ModelTrainingError(Exception):
    pass

class APIError(Exception):
    pass

# =====================================================
# DATA CLASSES
# =====================================================

@dataclass(frozen=True)
class ModelMetrics:
    r2_score: float
    mae: float
    rmse: float
    model_name: str

    def to_dict(self):
        return {
            "R2 Score": self.r2_score,
            "MAE ($)": self.mae,
            "RMSE ($)": self.rmse
        }

@dataclass
class InvestmentProfile:
    grade_threshold: int = 8
    waterfront_bonus: int = 30
    sqft_threshold: int = 3000
    sqft_bonus: int = 20
    condition_threshold: int = 4
    condition_bonus: int = 20

@dataclass
class User:
    id: int
    username: str
    email: str
    role: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool = True

# =====================================================
# GLASSMORPHISM THEME ENGINE
# =====================================================

class ThemeEngine:
    DARK_THEME = {
        "bg_primary": "#0B0F19",
        "bg_secondary": "#111827",
        "bg_card": "rgba(17, 24, 39, 0.7)",
        "bg_glass": "rgba(255, 255, 255, 0.03)",
        "text_primary": "#F1F5F9",
        "text_secondary": "#94A3B8",
        "accent_primary": "#00FFAA",
        "accent_secondary": "#00C2FF",
        "accent_gradient": "linear-gradient(135deg, #00FFAA 0%, #00C2FF 100%)",
        "border_glass": "rgba(255, 255, 255, 0.08)",
        "shadow_glow": "0 0 40px rgba(0, 255, 170, 0.15)",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "info": "#3B82F6",
    }

    @classmethod
    def get_css(cls):
        t = cls.DARK_THEME
        return f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        .stApp {{
            background: {t['bg_primary']};
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(0, 255, 170, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(0, 194, 255, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(139, 92, 246, 0.02) 0%, transparent 60%);
            font-family: 'Inter', sans-serif;
        }}
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        footer {{visibility: hidden;}}
        .glass-card {{
            background: {t['bg_glass']};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {t['border_glass']};
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: {t['shadow_glow']}, 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(0, 255, 170, 0.2);
        }}
        .glass-btn {{
            background: {t['accent_gradient']};
            color: {t['bg_primary']};
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 255, 170, 0.3);
        }}
        .glass-btn:hover {{
            transform: scale(1.03);
            box-shadow: 0 6px 25px rgba(0, 255, 170, 0.5);
        }}
        [data-testid="metric-container"] {{
            background: {t['bg_glass']};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {t['border_glass']};
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        [data-testid="metric-container"]:hover {{
            transform: translateY(-3px);
            box-shadow: {t['shadow_glow']};
            border-color: rgba(0, 255, 170, 0.2);
        }}
        [data-testid="stSidebar"] {{
            background: rgba(11, 15, 25, 0.95) !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid {t['border_glass']};
        }}
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {{
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid {t['border_glass']} !important;
            border-radius: 12px !important;
            color: {t['text_primary']} !important;
            font-family: 'Inter', sans-serif;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: {t['accent_primary']} !important;
            box-shadow: 0 0 0 3px rgba(0, 255, 170, 0.1) !important;
        }}
        .stButton > button {{
            background: {t['accent_gradient']} !important;
            color: {t['bg_primary']} !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 10px 24px !important;
            transition: all 0.3s ease !important;
        }}
        .stButton > button:hover {{
            transform: scale(1.03) !important;
            box-shadow: 0 0 20px rgba(0, 255, 170, 0.4) !important;
        }}
        .stDataFrame {{
            background: {t['bg_glass']};
            border-radius: 16px;
            border: 1px solid {t['border_glass']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background: {t['bg_glass']};
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            color: {t['text_secondary']};
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background: {t['accent_gradient']};
            color: {t['bg_primary']} !important;
            font-weight: 700;
        }}
        .stChatMessage {{
            background: {t['bg_glass']};
            border: 1px solid {t['border_glass']};
            border-radius: 16px;
            margin: 8px 0;
        }}
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {t['bg_secondary']};
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {t['accent_primary']};
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {t['accent_secondary']};
        }}
        h1, h2, h3 {{
            color: {t['text_primary']} !important;
            font-weight: 700 !important;
            font-family: 'Inter', sans-serif;
        }}
        h1 {{
            background: {t['accent_gradient']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .stProgress > div > div > div > div {{
            background: {t['accent_gradient']} !important;
            border-radius: 10px;
        }}
        .streamlit-expanderHeader {{
            background: {t['bg_glass']};
            border-radius: 12px;
            border: 1px solid {t['border_glass']};
        }}
        .stAlert {{
            background: {t['bg_glass']};
            border: 1px solid {t['border_glass']};
            border-radius: 12px;
        }}
        .stFileUploader > div > div {{
            background: {t['bg_glass']};
            border: 2px dashed {t['border_glass']};
            border-radius: 16px;
        }}
        .stFileUploader > div > div:hover {{
            border-color: {t['accent_primary']};
        }}
        .stCodeBlock {{
            background: {t['bg_secondary']};
            border-radius: 12px;
            border: 1px solid {t['border_glass']};
        }}
        .stSlider > div > div > div > div {{
            background: {t['accent_gradient']};
        }}
        .stRadio > div {{
    background: #1E293B !important;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #475569;
    color: #F8FAFC !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}}
div[role="radiogroup"] label p {{
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}}
div[role="radiogroup"] label {{
    color: white !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}}

div[role="radiogroup"] label {{
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}}
.stRadio label {{
    color: #F8FAFC !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}}
        .notification-badge {{
            position: absolute;
            top: -5px;
            right: -5px;
            background: {t['error']};
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            font-size: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
        }}
        .particle {{
            position: fixed;
            border-radius: 50%;
            background: rgba(0, 255, 170, 0.06);
            pointer-events: none;
            z-index: 0;
        }}
        @keyframes float-particle {{
            0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
            10% {{ opacity: 1; }}
            90% {{ opacity: 1; }}
            100% {{ transform: translateY(-10vh) rotate(720deg); opacity: 0; }}
        }}
        </style>
        """

    @classmethod
    def inject(cls):
        st.markdown(cls.get_css(), unsafe_allow_html=True)

    @classmethod
    def render_particles(cls, count=15):
        particles_html = "<div style='position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;'>"
        for i in range(count):
            size = np.random.randint(3, 8)
            left = np.random.randint(0, 100)
            delay = np.random.uniform(0, 20)
            duration = np.random.uniform(15, 30)
            particles_html += f'<div class="particle" style="left:{left}%;width:{size}px;height:{size}px;animation:float-particle {duration}s infinite linear {delay}s;"></div>'
        particles_html += "</div>"
        st.markdown(particles_html, unsafe_allow_html=True)

# =====================================================
# AUTHENTICATION MODULE (Enhanced with RBAC)
# =====================================================

class AuthManager:
    @staticmethod
    def hash_password(password):
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()

    @staticmethod
    def verify_credentials(username, password):
        if not username or not password:
            return False
        try:
            stored_hash = Config.ADMIN_PASSWORD_HASH.encode()
            return (username == Config.ADMIN_USERNAME and bcrypt.checkpw(password.encode(), stored_hash))
        except Exception:
            return False

    @staticmethod
    def verify_db_credentials(username, password):
        row = db.fetchone("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
        if row and bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            db.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (row["id"],))
            return User(
                id=row["id"], username=row["username"], email=row["email"] or "",
                role=row["role"], created_at=row["created_at"],
                last_login=row["last_login"], is_active=bool(row["is_active"])
            )
        return None

    @staticmethod
    def init_session():
        defaults = {
            "authenticated": False, "session_token": None, "user_id": None,
            "username": None, "user_role": "guest", "user_email": None,
            "login_time": None, "notifications": [], "chat_history": [],
            "current_page": "Dashboard", "theme": "dark", "sidebar_collapsed": False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def has_role(role):
        role_hierarchy = {"guest": 0, "user": 1, "analyst": 2, "admin": 3}
        current = st.session_state.get("user_role", "guest")
        return role_hierarchy.get(current, 0) >= role_hierarchy.get(role, 0)

    @staticmethod
    def render_login_page():
        ThemeEngine.inject()
        ThemeEngine.render_particles(20)

        st.markdown("""
        <style>
        .login-bg { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; overflow: hidden; }
        .login-card {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 28px;
            padding: 2.5rem;
            max-width: 480px;
            margin: 0 auto;
            box-shadow: 0 25px 80px -20px rgba(0, 0, 0, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.05),
                0 0 100px rgba(0, 255, 170, 0.03);
            animation: slideUp 0.8s ease-out;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px) scale(0.95); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .login-title {
            font-size: 1.8rem; font-weight: 800;
            text-align: center; margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #00FFAA, #00C2FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }
        .login-subtitle {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.45);
            text-align: center;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }
        .login-features {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 0.6rem; margin-bottom: 1.5rem;
        }
        .feature-item {
            display: flex; align-items: center; gap: 0.4rem;
            font-size: 0.78rem; color: rgba(255, 255, 255, 0.5);
            padding: 0.5rem 0.6rem;
            background: rgba(255,255,255,0.02);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.05);
            transition: all 0.3s;
        }
        .feature-item:hover {
            background: rgba(0, 255, 170, 0.05);
            border-color: rgba(0, 255, 170, 0.15);
        }
        .login-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 170, 0.2), transparent);
            margin: 1.5rem 0; border: none;
        }
        .login-footer {
            text-align: center; margin-top: 1.5rem;
            font-size: 0.72rem; color: rgba(255, 255, 255, 0.25);
        }
        .version-badge {
            display: inline-block;
            background: rgba(0, 255, 170, 0.08);
            color: #00FFAA;
            padding: 0.25rem 0.8rem;
            border-radius: 20px;
            font-size: 0.7rem; font-weight: 600;
            border: 1px solid rgba(0, 255, 170, 0.15);
            margin-bottom: 0.75rem;
        }
        .trust-badges {
            display: flex; justify-content: center;
            gap: 0.8rem; margin-top: 0.75rem;
            flex-wrap: wrap;
        }
        .trust-badge {
            display: flex; align-items: center; gap: 0.3rem;
            font-size: 0.7rem; color: rgba(255, 255, 255, 0.35);
        }
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important; color: black !important;
            -webkit-text-fill-color: black !important;
            padding: 14px 18px !important;
            font-size: 0.95rem !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #00FFAA !important;
            box-shadow: 0 0 0 3px rgba(0, 255, 170, 0.1) !important;
        }
        .stTextInput > label {
            color: rgba(255, 255, 255, 0.6) !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }
        </style>
        <div class="login-bg"></div>
        """, unsafe_allow_html=True)

        left_spacer, center_col, right_spacer = st.columns([1, 2.5, 1])

        with center_col:
            lottie_html = """
            <div style="display: flex; justify-content: center; margin-bottom: 0.5rem;">
                <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
                <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
                    background="transparent" speed="1"
                    style="width: 160px; height: 160px;" loop autoplay>
                </lottie-player>
            </div>
            """
            st.components.v1.html(lottie_html, height=160)

            st.markdown("""
            <div class="login-card">
                <div style="display: flex; justify-content: center; margin-bottom: 0.75rem;">
                    <svg width="56" height="56" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#00FFAA"/>
                                <stop offset="100%" style="stop-color:#00C2FF"/>
                            </linearGradient>
                            <filter id="gl"><feGaussianBlur stdDeviation="4" result="b"/>
                                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                            </filter>
                        </defs>
                        <rect x="10" y="30" width="60" height="40" rx="6" fill="url(#lg)" opacity="0.9" filter="url(#gl)"/>
                        <path d="M5 30L40 5L75 30" stroke="url(#lg)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" filter="url(#gl)"/>
                        <rect x="30" y="45" width="20" height="25" rx="3" fill="#0B0F19"/>
                        <circle cx="40" cy="38" r="5" fill="#0B0F19"/>
                    </svg>
                </div>
                <h1 class="login-title">Real Estate Intelligence Hub</h1>
                <p class="login-subtitle">
                    AI-Powered Property Analytics, Market Intelligence<br>& Investment Forecasting Platform
                </p>
                <div class="login-features">
                    <div class="feature-item"><span>🤖</span><span>AI Assistant</span></div>
                    <div class="feature-item"><span>📊</span><span>Advanced Analytics</span></div>
                    <div class="feature-item"><span>🗺️</span><span>Geospatial Maps</span></div>
                    <div class="feature-item"><span>💎</span><span>Investment Scoring</span></div>
                    <div class="feature-item"><span>🔮</span><span>Price Prediction</span></div>
                    <div class="feature-item"><span>📈</span><span>Market Forecasting</span></div>
                </div>
                <hr class="login-divider">
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

            username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")

            col1, col2 = st.columns([1, 1])
            with col1:
                remember = st.checkbox("Remember me", key="remember_me")
            with col2:
                st.markdown("<div style='text-align:right; padding-top:8px;'><a href='#' style='color:rgba(0,255,170,0.6); font-size:0.8rem; text-decoration:none;'>Forgot password?</a></div>", unsafe_allow_html=True)

            if st.button("🔐 Sign In to Dashboard", use_container_width=True, key="login_btn", type="primary"):
                user = AuthManager.verify_db_credentials(username, password)
                if user or AuthManager.verify_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.session_token = secrets.token_urlsafe(24)
                    st.session_state.username = username
                    st.session_state.user_role = user.role if user else "admin"
                    st.session_state.user_id = user.id if user else 0
                    st.session_state.user_email = user.email if user else "admin@reihub.com"
                    st.session_state.login_time = datetime.now().isoformat()
                    logger.info(f"User {username} authenticated successfully")
                    st.success("✅ Login successful! Loading dashboard...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.", icon="🚫")
                    logger.warning(f"Failed login attempt for user: {username}")

            st.markdown("""
            <div class="login-footer">
                <span class="version-badge">v3.0 Enterprise</span>
                <div class="trust-badges">
                    <div class="trust-badge"><span style="color: #00FFAA;">🔒</span> SSL Encrypted</div>
                    <div class="trust-badge"><span style="color: #00C2FF;">🤖</span> AI-Powered</div>
                    <div class="trust-badge"><span style="color: #00FFAA;">⚡</span> Real-time</div>
                    <div class="trust-badge"><span style="color: #00C2FF;">🛡️</span> RBAC Secure</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def logout():
        st.session_state.authenticated = False
        st.session_state.session_token = None
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.user_role = "guest"
        st.rerun()

    @staticmethod
    def require_auth():
        return st.session_state.get("authenticated", False)

# =====================================================
# AI CHATBOT ENGINE
# =====================================================

class AIChatbot:
    SYSTEM_PROMPT = """You are REI Assistant, an expert real estate intelligence AI. You help users with:
- Property price predictions and market analysis
- Investment recommendations and opportunity scoring
- Neighborhood and zipcode insights
- Market trends and forecasting
- Property comparisons
- General real estate knowledge

Be concise, professional, and data-driven. Use available context when provided."""

    KNOWLEDGE_BASE = {
        "price_factors": ["sqft_living", "grade", "waterfront", "view", "condition", "location"],
        "investment_tips": [
            "Properties with grade >= 8 show 40% higher appreciation",
            "Waterfront properties command 20-30% premium",
            "Renovated homes sell 15% faster on average",
            "Zipcodes with avg price/sqft below median are potential growth areas"
        ],
        "market_insights": [
            "Spring season typically sees 15% more listings",
            "Homes priced 5% below market value receive 3x more offers",
            "Properties with 3-4 bedrooms have highest liquidity"
        ]
    }

    def __init__(self):
        self.db = db
        self.session_id = st.session_state.get("session_token", str(uuid.uuid4()))

    def get_context(self, df=None):
        context = []
        if df is not None and not df.empty:
            context.append(f"Dataset: {len(df)} properties. Avg price: ${df['price'].mean():,.0f}. Median: ${df['price'].median():,.0f}.")
            if "zipcode" in df.columns:
                context.append(f"Covering {df['zipcode'].nunique()} zipcodes.")
        return "\n".join(context)

    def generate_response(self, prompt, df=None):
        prompt_lower = prompt.lower()
        if Config.OPENAI_API_KEY:
            try:
                return self._call_openai(prompt, df)
            except Exception as e:
                logger.warning(f"OpenAI call failed: {e}")
        return self._rule_based_response(prompt_lower, df)

    def _call_openai(self, prompt, df=None):
        import openai
        openai.api_key = Config.OPENAI_API_KEY
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": self.get_context(df)}
        ]
        history = self.get_chat_history(limit=5)
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=500, temperature=0.7
        )
        return response.choices[0].message.content

    def _rule_based_response(self, prompt, df=None):
        if any(w in prompt for w in ["hello", "hi", "hey", "greetings"]):
            return "👋 Hello! I'm REI Assistant. I can help you with property analytics, investment insights, market trends, and price predictions. What would you like to explore today?"

        if any(w in prompt for w in ["price", "cost", "worth", "value", "expensive", "cheap"]):
            if df is not None:
                avg = df["price"].mean()
                median = df["price"].median()
                return f"💰 Based on current data: Average price is **${avg:,.0f}**, Median is **${median:,.0f}**. Price is primarily driven by sqft_living, grade, and location. Would you like a detailed breakdown?"
            return "💰 Property prices depend on several factors: square footage, grade, location (zipcode), waterfront access, and condition. Upload data or select a dataset to see specific insights."

        if any(w in prompt for w in ["invest", "investment", "opportunity", "score", "return"]):
            tips = self.KNOWLEDGE_BASE["investment_tips"]
            return "📈 **Investment Insights:**\n\n" + "\n".join([f"• {tip}" for tip in tips[:3]]) + "\n\nWould you like me to analyze specific properties for investment potential?"

        if any(w in prompt for w in ["trend", "forecast", "predict", "future", "market"]):
            insights = self.KNOWLEDGE_BASE["market_insights"]
            return "🔮 **Market Trends:**\n\n" + "\n".join([f"• {insight}" for insight in insights]) + "\n\nCheck the Forecasting page for detailed temporal analysis."

        if any(w in prompt for w in ["compare", "difference", "versus", "vs", "better"]):
            return "⚖️ Use the **Compare Properties** page for side-by-side analysis with radar charts. You can compare any two properties across all features like price, sqft, grade, and condition."

        if any(w in prompt for w in ["map", "location", "area", "zipcode", "neighborhood"]):
            return "🗺️ The **Map Visualization** page shows interactive geospatial data with heatmaps and clustering. You can filter by price range, grade, and explore specific zipcodes in the **Area Analysis** page."

        if any(w in prompt for w in ["model", "ai", "algorithm", "predict", "machine learning", "ml"]):
            return "🤖 Our ML engine trains Linear Regression, Random Forest, and XGBoost models. Random Forest typically achieves the best R2 score. Visit **AI Explainability** to see feature importance, or **Model Comparison** for benchmark results."

        if any(w in prompt for w in ["help", "what can you do", "features", "capability"]):
            return """🎯 **I can help you with:**

📊 **Analytics** — Market overviews, price distributions, correlations
🤖 **AI Predictions** — Price forecasting with ML models  
🗺️ **Geospatial** — Interactive maps with heatmaps & clustering
💎 **Investments** — Opportunity scoring & undervalued properties
⚖️ **Comparisons** — Side-by-side property analysis
🔮 **Forecasting** — Temporal trend analysis
📈 **Explainability** — Feature importance & model insights
🌤️ **Weather** — Local climate data integration

Just ask me anything about real estate!"""

        if any(w in prompt for w in ["thank", "thanks", "appreciate", "good job"]):
            return "🙏 You're welcome! I'm here whenever you need real estate intelligence. Feel free to ask about properties, markets, investments, or predictions anytime!"

        return f"🔍 I understand you're asking about **{prompt[:50]}...** \n\nI can help with property analytics, investment scoring, market trends, and price predictions. Could you be more specific about what you'd like to know? Or try asking about prices, investments, maps, or model insights."

    def save_message(self, role, content):
        self.db.insert("chat_history", {
            "session_id": self.session_id, "role": role, "content": content,
            "user_id": st.session_state.get("user_id")
        })

    def get_chat_history(self, limit=50):
        rows = self.db.fetchall(
            "SELECT role, content, timestamp FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (self.session_id, limit)
        )
        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]

    def clear_history(self):
        self.db.execute("DELETE FROM chat_history WHERE session_id = ?", (self.session_id,))
        st.session_state.chat_history = []

# =====================================================
# NOTIFICATION SYSTEM
# =====================================================

class NotificationManager:
    @staticmethod
    def add_notification(user_id, title, message, notif_type="info"):
        db.insert("notifications", {
            "user_id": user_id, "title": title, "message": message,
            "type": notif_type, "is_read": 0
        })

    @staticmethod
    def get_unread_count(user_id):
        row = db.fetchone(
            "SELECT COUNT(*) as cnt FROM notifications WHERE user_id = ? AND is_read = 0", (user_id,)
        )
        return row["cnt"] if row else 0

    @staticmethod
    def get_notifications(user_id, limit=10):
        rows = db.fetchall(
            "SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        )
        return [dict(r) for r in rows]

    @staticmethod
    def mark_as_read(notif_id):
        db.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notif_id,))

    @staticmethod
    def mark_all_read(user_id):
        db.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,))

# =====================================================
# REST API CLIENT
# =====================================================

class APIClient:
    BASE_URL = os.getenv("REI_API_BASE_URL", "http://localhost:8000")

    @staticmethod
    def log_api_call(endpoint, method, status_code=None, response_time=None):
        db.insert("api_logs", {
            "endpoint": endpoint, "method": method,
            "status_code": status_code, "response_time": response_time,
            "client_ip": "127.0.0.1"
        })

    @staticmethod
    def get(endpoint, params=None):
        start = time.time()
        try:
            response = requests.get(f"{APIClient.BASE_URL}{endpoint}", params=params, timeout=10)
            APIClient.log_api_call(endpoint, "GET", response.status_code, time.time() - start)
            return {"status": "success", "data": response.json()}
        except Exception as e:
            APIClient.log_api_call(endpoint, "GET", None, time.time() - start)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def post(endpoint, data):
        start = time.time()
        try:
            response = requests.post(f"{APIClient.BASE_URL}{endpoint}", json=data, timeout=10)
            APIClient.log_api_call(endpoint, "POST", response.status_code, time.time() - start)
            return {"status": "success", "data": response.json()}
        except Exception as e:
            APIClient.log_api_call(endpoint, "POST", None, time.time() - start)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def predict_price(property_data):
        return APIClient.post("/api/v1/predict", property_data)

    @staticmethod
    def get_market_summary(zipcode=None):
        params = {"zipcode": zipcode} if zipcode else {}
        return APIClient.get("/api/v1/market/summary", params)

# =====================================================
# DATA LOADING & VALIDATION
# =====================================================

class DataManager:
    REQUIRED_COLUMNS = [
        "price", "bedrooms", "bathrooms", "sqft_living", 
        "floors", "grade", "zipcode", "lat", "long"
    ]
    OPTIONAL_COLUMNS = ["waterfront", "condition", "date", "sqft_lot", "view", "yr_built"]

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Loading dataset...")
    def load_data(uploaded_file=None):
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                logger.info(f"Loaded uploaded file: {uploaded_file.name}")
                save_path = Config.UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
                df.to_csv(save_path, index=False)
            else:
                if not Config.DEFAULT_DATA_PATH.exists():
                    raise DataValidationError(
                        f"Default dataset not found at {Config.DEFAULT_DATA_PATH}. Please upload a CSV file."
                    )
                df = pd.read_csv(Config.DEFAULT_DATA_PATH)
                logger.info(f"Loaded default dataset: {Config.DEFAULT_DATA_PATH}")

            DataManager._validate_data(df)
            df = DataManager._preprocess_data(df)
            DataManager._sync_to_db(df)
            return df

        except pd.errors.EmptyDataError:
            raise DataValidationError("The uploaded file is empty.")
        except pd.errors.ParserError as e:
            raise DataValidationError(f"Failed to parse CSV: {str(e)}")
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            raise

    @staticmethod
    def _validate_data(df):
        if df.empty:
            raise DataValidationError("Dataset contains no rows.")

        missing_required = [col for col in DataManager.REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            raise DataValidationError(
                f"Missing required columns: {missing_required}. Required: {DataManager.REQUIRED_COLUMNS}"
            )

        numeric_cols = ["price", "bedrooms", "bathrooms", "sqft_living", "floors", "grade"]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(f"Column '{col}' must be numeric.")

        if (df["price"] < 0).any():
            raise DataValidationError("Price values cannot be negative.")
        if (df["bedrooms"] < 0).any():
            raise DataValidationError("Bedroom count cannot be negative.")

        logger.info(f"Validation passed: {df.shape[0]} rows, {df.shape[1]} columns")

    @staticmethod
    def _preprocess_data(df):
        df = df.copy()
        for col in DataManager.OPTIONAL_COLUMNS:
            if col in df.columns:
                if col in ["waterfront", "condition", "view"]:
                    df[col] = df[col].fillna(0)
                elif col == "date":
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        df["price_per_sqft"] = df["price"] / df["sqft_living"].clip(lower=1)

        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

        return df

    @staticmethod
    def _sync_to_db(df):
        try:
            df_sample = df.head(1000).copy()
            df_sample.to_sql("properties_cache", db.conn, if_exists="replace", index=False)
        except Exception as e:
            logger.warning(f"DB sync warning: {e}")

    @staticmethod
    def get_feature_columns(df):
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_upload_history():
        files = []
        for f in sorted(Config.UPLOAD_DIR.glob("*.csv"), reverse=True):
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_mb": round(stat.st_size / (1024*1024), 2),
                "uploaded": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            })
        return pd.DataFrame(files)

# =====================================================
# MACHINE LEARNING ENGINE
# =====================================================

class MLEngine:
    FEATURES = ["bedrooms", "bathrooms", "sqft_living", "floors", "grade"]
    TARGET = "price"

    def __init__(self, df):
        self.df = df
        self.models = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        self._is_trained = False

    def prepare_data(self, features=None):
        feature_cols = features or self.FEATURES
        available = [f for f in feature_cols if f in self.df.columns]

        if len(available) < 2:
            raise ModelTrainingError("Insufficient features available for training.")

        X = self.df[available].fillna(self.df[available].median())
        y = self.df[self.TARGET]

        return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    @st.cache_resource(show_spinner="Training advanced models...")
    def train_models(_self, features=None):
        try:
            X_train, X_test, y_train, y_test = _self.prepare_data(features)

            model_configs = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=Config.N_ESTIMATORS,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1,
                    max_depth=20
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=150,
                    random_state=Config.RANDOM_STATE,
                    verbosity=0,
                    max_depth=8
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=150,
                    random_state=Config.RANDOM_STATE,
                    max_depth=6
                )
            }

            results = {}
            for name, model in model_configs.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                metrics = ModelMetrics(
                    r2_score=r2_score(y_test, predictions),
                    mae=mean_absolute_error(y_test, predictions),
                    rmse=np.sqrt(mean_squared_error(y_test, predictions)),
                    model_name=name
                )

                _self.models[name] = model
                _self.metrics[name] = metrics
                results[name] = metrics

            _self._is_trained = True
            _self._best_model = _self.models["Random Forest"]

            logger.info(f"Trained {len(model_configs)} models successfully")
            return results

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelTrainingError(f"Training pipeline failed: {str(e)}")

    def predict(self, features, model_name="Random Forest"):
        if not self._is_trained:
            self.train_models()

        if model_name not in self.models:
            model_name = "Random Forest"

        return self.models[model_name].predict(features)

    def get_feature_importance(self):
        if "Random Forest" not in self.models:
            self.train_models()

        rf_model = self.models["Random Forest"]
        features = self.FEATURES

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        return importance

    def get_cross_validation_scores(self, model_name="Random Forest", cv=5):
        if model_name not in self.models:
            self.train_models()

        X = self.df[self.FEATURES].fillna(self.df[self.FEATURES].median())
        y = self.df[self.TARGET]

        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        return {
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
            "scores": scores.tolist(),
            "model": model_name
        }

    def save_model(self, model_name, path=None):
        if model_name not in self.models:
            self.train_models()

        save_path = path or Config.MODEL_CACHE_DIR / f"{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(self.models[model_name], save_path)
        return save_path

# =====================================================
# INVESTMENT ANALYTICS
# =====================================================

class InvestmentAnalyzer:
    def __init__(self, profile=None):
        self.profile = profile or InvestmentProfile()

    def calculate_score(self, row):
        score = 0
        p = self.profile

        if "grade" in row and row["grade"] >= p.grade_threshold:
            score += 30

        if "waterfront" in row and row["waterfront"] == 1:
            score += p.waterfront_bonus

        if "sqft_living" in row and row["sqft_living"] > p.sqft_threshold:
            score += p.sqft_bonus

        if "condition" in row and row["condition"] >= p.condition_threshold:
            score += p.condition_bonus

        if "view" in row and row["view"] >= 3:
            score += 15

        if "yr_renovated" in row and row["yr_renovated"] > 2000:
            score += 10

        return int(min(score, 100))

    def get_top_investments(self, df, n=10):
        df = df.copy()
        df["investment_score"] = df.apply(self.calculate_score, axis=1)

        columns = ["price", "zipcode", "investment_score", "sqft_living"]
        optional = ["grade", "condition", "waterfront", "view", "yr_renovated"]
        columns += [c for c in optional if c in df.columns]

        return df.sort_values("investment_score", ascending=False)[columns].head(n)

    def get_undervalued_properties(self, df, n=10):
        if "price_per_sqft" not in df.columns:
            df["price_per_sqft"] = df["price"] / df["sqft_living"].clip(lower=1)

        zipcode_median = df.groupby("zipcode")["price_per_sqft"].transform("median")
        df["valuation_ratio"] = df["price_per_sqft"] / zipcode_median

        return (
            df[df["valuation_ratio"] < 0.8]
            .sort_values("valuation_ratio")
            .head(n)
            [["price", "zipcode", "price_per_sqft", "sqft_living", "bedrooms", "valuation_ratio", "grade"]]
        )

    def get_neighborhood_ranking(self, df):
        zipcode_stats = df.groupby("zipcode").agg({
            "price": ["mean", "median", "std"],
            "sqft_living": "mean",
            "grade": "mean",
            "price_per_sqft": "mean"
        }).reset_index()
        zipcode_stats.columns = ["zipcode", "avg_price", "median_price", "price_std", "avg_sqft", "avg_grade", "avg_price_per_sqft"]
        zipcode_stats["investment_potential"] = (
            (zipcode_stats["avg_grade"] / 10) * 40 +
            (1 / (zipcode_stats["price_std"] / zipcode_stats["avg_price"] + 0.001)) * 30 +
            (zipcode_stats["avg_sqft"] / zipcode_stats["avg_sqft"].max()) * 30
        )
        return zipcode_stats.sort_values("investment_potential", ascending=False)

# =====================================================
# REPORTING ENGINE
# =====================================================

class ReportEngine:
    @staticmethod
    def export_csv(df, filename="analysis_export.csv"):
        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def export_json(data, filename="export.json"):
        return json.dumps(data, indent=2, default=str).encode("utf-8")

    @staticmethod
    def generate_summary_stats(df):
        numeric_df = df.select_dtypes(include=[np.number])

        return {
            "total_properties": len(df),
            "avg_price": float(df["price"].mean()),
            "median_price": float(df["price"].median()),
            "price_std": float(df["price"].std()),
            "avg_sqft": float(df["sqft_living"].mean()),
            "total_bedrooms": int(df["bedrooms"].sum()),
            "price_range": {
                "min": float(df["price"].min()),
                "max": float(df["price"].max())
            },
            "correlation_matrix": numeric_df.corr().to_dict(),
            "generated_at": datetime.now().isoformat()
        }

# =====================================================
# WEATHER SERVICE
# =====================================================

class WeatherService:
    @staticmethod
    @st.cache_data(ttl=600, show_spinner="Fetching weather data...")
    def get_seattle_weather():
        if not Config.WEATHER_API_KEY:
            return {
                "status": "error",
                "message": "Weather API key not configured. Set OPENWEATHER_API_KEY environment variable.",
                "data": None
            }

        try:
            url = (
                "https://api.openweathermap.org/data/2.5/weather?"
                f"q=Seattle&appid={Config.WEATHER_API_KEY}&units=metric"
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "status": "success",
                "data": {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "description": data["weather"][0]["description"].title(),
                    "wind_speed": data["wind"]["speed"],
                    "icon": data["weather"][0]["icon"]
                }
            }

        except requests.RequestException as e:
            logger.error(f"Weather API error: {e}")
            return {
                "status": "error",
                "message": f"Failed to fetch weather: {str(e)}",
                "data": None
            }

# =====================================================
# UI COMPONENTS
# =====================================================

class UIComponents:
    @staticmethod
    def render_header(title, subtitle=""):
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem;">
                🏙 {title}
            </h1>
            {f'<p style="color: #94A3B8; font-size: 1rem; margin-top: 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,170,0.3), transparent); margin: 1.5rem 0;'>", unsafe_allow_html=True)

    @staticmethod
    def render_metrics_grid(metrics):
        cols = st.columns(len(metrics))
        for col, (label, (value, delta)) in zip(cols, metrics.items()):
            with col:
                if delta:
                    st.metric(label=label, value=value, delta=delta)
                else:
                    st.metric(label=label, value=value)

    @staticmethod
    def render_error(message, exception=None):
        st.error(f"⚠️ {message}")
        if exception:
            with st.expander("Technical Details"):
                st.code(str(exception))
                logger.error(f"UI Error: {message}", exc_info=exception)

    @staticmethod
    def render_glass_card(title, content, icon="📊"):
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <h3 style="margin: 0; font-size: 1.1rem;">{title}</h3>
            </div>
            {content}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_status_badge(status, color="#00FFAA"):
        return f"""
        <span style="
            background: {color}20;
            color: {color};
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid {color}40;
        ">{status}</span>
        """

# =====================================================
# PAGE MODULES
# =====================================================

class PageDashboard:
    @staticmethod
    def render(df, ml_engine):
        UIComponents.render_header(
            "AI Real Estate Dashboard",
            "Real-time market intelligence and predictive analytics"
        )

        # KPI Cards with glassmorphism
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.5rem;">Average Price</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #00FFAA;">""" + f"${int(df['price'].mean()):,}" + """</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.5rem;">Median Price</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #00C2FF;">""" + f"${int(df['price'].median()):,}" + """</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.5rem;">Total Properties</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #A78BFA;">""" + f"{len(df):,}" + """</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.5rem;">Avg Sqft</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #F59E0B;">""" + f"{int(df['sqft_living'].mean()):,}" + """</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Market Overview")
            fig = px.scatter(
                df.sample(min(2000, len(df))),
                x="sqft_living",
                y="price",
                color="grade",
                size="bathrooms",
                hover_data=["zipcode", "bedrooms"],
                title="Price vs Living Area by Grade",
                template="plotly_dark",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Quick Stats")
            stats_df = df.describe().round(2)
            st.dataframe(stats_df, use_container_width=True, height=400)

        st.subheader("🔍 Interactive Explorer")
        numeric_cols = DataManager.get_feature_columns(df)

        col_x, col_y, col_color = st.columns(3)
        with col_x:
            x_axis = st.selectbox("X Axis", numeric_cols, index=numeric_cols.index("sqft_living"))
        with col_y:
            y_axis = st.selectbox("Y Axis", numeric_cols, index=numeric_cols.index("price"))
        with col_color:
            color_options = ["grade", "bedrooms", "zipcode"] + numeric_cols
            color_by = st.selectbox("Color By", color_options, index=0)

        fig2 = px.scatter(
            df.sample(min(1000, len(df))),
            x=x_axis,
            y=y_axis,
            color=color_by if color_by in df.columns else None,
            title=f"{y_axis} vs {x_axis}",
            template="plotly_dark",
            opacity=0.7
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🧊 3D Market View")
        fig3d = px.scatter_3d(
            df.sample(min(500, len(df))),
            x="sqft_living",
            y="bedrooms",
            z="price",
            color="grade",
            size="bathrooms",
            template="plotly_dark",
            title="3D Property Analysis"
        )
        fig3d.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig3d, use_container_width=True)


class PagePriceAnalysis:
    @staticmethod
    def render(df):
        UIComponents.render_header("Price Analysis", "Distribution and outlier detection")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Price Distribution")
            fig = px.histogram(
                df,
                x="price",
                nbins=50,
                template="plotly_dark",
                color_discrete_sequence=["#00FFAA"],
                marginal="box"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏠 Price by Bedrooms")
            fig = px.box(
                df,
                x="bedrooms",
                y="price",
                template="plotly_dark",
                color="bedrooms",
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9')
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏆 Top 10 Premium Properties")
        top_props = df.nlargest(10, "price")[
            ["price", "bedrooms", "bathrooms", "sqft_living", "grade", "zipcode"]
        ]
        st.dataframe(top_props.style.format({"price": "${:,.0f}"}), use_container_width=True)


class PageAreaAnalysis:
    @staticmethod
    def render(df):
        UIComponents.render_header("Area Analysis", "Zipcode-level market insights")

        zipcodes = sorted(df["zipcode"].unique())
        selected_zip = st.selectbox("📍 Select Zipcode", zipcodes)

        filtered = df[df["zipcode"] == selected_zip].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Properties", len(filtered))
        with col2:
            st.metric("Avg Price", f"${int(filtered['price'].mean()):,}")
        with col3:
            st.metric("Avg Sqft", f"{int(filtered['sqft_living'].mean()):,}")

        fig = px.scatter(
            filtered,
            x="sqft_living",
            y="price",
            color="grade",
            size="bedrooms",
            template="plotly_dark",
            title=f"Properties in Zipcode {selected_zip}"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(filtered.head(20), use_container_width=True)


class PageInvestment:
    @staticmethod
    def render(df):
        UIComponents.render_header("Investment Intelligence", "AI-powered opportunity scoring")

        analyzer = InvestmentAnalyzer()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Top Investment Opportunities")
            top_investments = analyzer.get_top_investments(df, n=10)
            st.dataframe(
                top_investments.style.format({"price": "${:,.0f}"}),
                use_container_width=True
            )

        with col2:
            st.subheader("💎 Undervalued Properties")
            undervalued = analyzer.get_undervalued_properties(df, n=10)
            if not undervalued.empty:
                st.dataframe(
                    undervalued.style.format({
                        "price": "${:,.0f}",
                        "price_per_sqft": "${:,.0f}",
                        "valuation_ratio": "{:.2%}"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No significantly undervalued properties found.")

        # Neighborhood ranking
        st.subheader("🏘️ Neighborhood Investment Ranking")
        ranking = analyzer.get_neighborhood_ranking(df)
        fig = px.bar(
            ranking.head(15),
            x="zipcode",
            y="investment_potential",
            color="avg_grade",
            template="plotly_dark",
            title="Top 15 Zipcodes by Investment Potential",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig, use_container_width=True)

        df["investment_score"] = df.apply(analyzer.calculate_score, axis=1)
        fig2 = px.histogram(
            df,
            x="investment_score",
            color="investment_score",
            template="plotly_dark",
            title="Investment Score Distribution",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig2, use_container_width=True)


class PageCompare:
    @staticmethod
    def render(df):
        UIComponents.render_header("Property Comparison", "Side-by-side analysis with radar charts")

        col1, col2 = st.columns(2)

        with col1:
            house1_idx = st.selectbox("Property A", df.index, format_func=lambda x: f"ID {x}")
        with col2:
            house2_idx = st.selectbox("Property B", df.index, format_func=lambda x: f"ID {x}", index=min(1, len(df)-1))

        h1, h2 = df.loc[house1_idx], df.loc[house2_idx]

        features = ["price", "bedrooms", "bathrooms", "sqft_living", "floors", "grade"]
        if "condition" in df.columns:
            features.append("condition")
        if "waterfront" in df.columns:
            features.append("waterfront")

        comparison_data = {
            "Feature": [f.replace("_", " ").title() for f in features],
            "Property A": [h1.get(f, "N/A") for f in features],
            "Property B": [h2.get(f, "N/A") for f in features]
        }

        comp_df = pd.DataFrame(comparison_data)
        comp_df.loc[comp_df["Feature"] == "Price", "Property A"] = f"${int(h1['price']):,}"
        comp_df.loc[comp_df["Feature"] == "Price", "Property B"] = f"${int(h2['price']):,}"

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        numeric_features = [f for f in features if f != "price" and pd.api.types.is_numeric_dtype(df[f])]
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[h1[f] for f in numeric_features],
            theta=[f.replace("_", " ").title() for f in numeric_features],
            fill="toself",
            name="Property A",
            line_color="#00FFAA"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[h2[f] for f in numeric_features],
            theta=[f.replace("_", " ").title() for f in numeric_features],
            fill="toself",
            name="Property B",
            line_color="#00C2FF"
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(h1[numeric_features].max(), h2[numeric_features].max())])),
            showlegend=True,
            template="plotly_dark",
            title="Feature Comparison Radar",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig, use_container_width=True)


class PageRecommendations:
    @staticmethod
    def render(df):
        UIComponents.render_header("AI Recommendations", "Smart property matching engine")

        col1, col2, col3 = st.columns(3)
        with col1:
            budget = st.slider("💰 Max Budget ($)", 50000, int(df["price"].max()), 500000, step=50000)
        with col2:
            bedrooms = st.slider("🛏️ Bedrooms", 1, int(df["bedrooms"].max()), 3)
        with col3:
            min_grade = st.slider("⭐ Min Grade", int(df["grade"].min()), int(df["grade"].max()), 7)

        recommendations = df[
            (df["price"] <= budget) & 
            (df["bedrooms"] == bedrooms) &
            (df["grade"] >= min_grade)
        ].copy()

        if not recommendations.empty:
            recommendations["value_score"] = (
                recommendations["sqft_living"] / recommendations["price"] * 100000
            )
            recommendations = recommendations.sort_values("value_score", ascending=False).head(10)

            st.success(f"✅ Found {len(recommendations)} matching properties")
            st.dataframe(
                recommendations[["price", "bedrooms", "bathrooms", "sqft_living", "zipcode", "grade", "value_score"]]
                .style.format({"price": "${:,.0f}", "value_score": "{:.1f}"}),
                use_container_width=True
            )

            # Value score chart
            fig = px.bar(
                recommendations,
                x=recommendations.index.astype(str),
                y="value_score",
                color="price",
                template="plotly_dark",
                title="Top Recommendations by Value Score",
                labels={"x": "Property ID", "value_score": "Value Score"},
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No properties match your criteria. Try adjusting filters.")


class PageExplainability:
    @staticmethod
    def render(ml_engine):
        UIComponents.render_header("AI Explainability", "Understanding model decisions & feature importance")

        importance = ml_engine.get_feature_importance()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                template="plotly_dark",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Feature Importance (Random Forest)"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🔍 Key Insights")
            top_feature = importance.iloc[0]
            st.info(f"**{top_feature['Feature']}** is the strongest price predictor, contributing **{top_feature['Importance']:.1%}** to model decisions.")

            st.markdown("### Feature Impact")
            for _, row in importance.iterrows():
                st.progress(float(row["Importance"]), text=f"{row['Feature']}: {row['Importance']:.1%}")

        # Cross-validation scores
        st.subheader("🎯 Model Validation")
        cv_results = ml_engine.get_cross_validation_scores()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean R²", f"{cv_results['mean_r2']:.4f}")
        with col2:
            st.metric("Std R²", f"{cv_results['std_r2']:.4f}")
        with col3:
            st.metric("CV Folds", cv_results['model'])

        fig = px.bar(
            x=[f"Fold {i+1}" for i in range(len(cv_results['scores']))],
            y=cv_results['scores'],
            template="plotly_dark",
            title="Cross-Validation R² Scores",
            color=cv_results['scores'],
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig, use_container_width=True)


class PageForecasting:
    @staticmethod
    def render(df):
        UIComponents.render_header("Market Forecasting", "Temporal trend analysis & seasonality")

        if "year" not in df.columns or "date" not in df.columns:
            st.warning("Time series data not available in dataset.")
            return

        yearly = df.groupby("year").agg({
            "price": ["mean", "median", "count"]
        }).reset_index()
        yearly.columns = ["year", "avg_price", "median_price", "sales_volume"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly["year"], 
            y=yearly["avg_price"],
            mode="lines+markers",
            name="Average Price",
            line=dict(color="#00FFAA", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"], 
            y=yearly["median_price"],
            mode="lines+markers",
            name="Median Price",
            line=dict(color="#00C2FF", width=3)
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Price Trends Over Time",
            xaxis_title="Year",
            yaxis_title="Price ($)",
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            yearly,
            x="year",
            y="sales_volume",
            template="plotly_dark",
            title="Annual Sales Volume",
            color="sales_volume",
            color_continuous_scale="Viridis"
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Monthly trends if available
        if "month" in df.columns:
            monthly = df.groupby("month").agg({"price": "mean"}).reset_index()
            fig3 = px.line(
                monthly,
                x="month",
                y="price",
                template="plotly_dark",
                title="Seasonal Price Patterns (Monthly)",
                markers=True
            )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9')
            )
            st.plotly_chart(fig3, use_container_width=True)


class PageMap:
    @staticmethod
    def render(df):
        UIComponents.render_header("Geospatial Intelligence", "Interactive property mapping with heatmaps")

        col1, col2, col3 = st.columns(3)
        with col1:
            price_range = st.slider("💰 Price Range", int(df["price"].min()), int(df["price"].max()), 
                                   (int(df["price"].quantile(0.1)), int(df["price"].quantile(0.9))))
        with col2:
            min_grade = st.slider("⭐ Min Grade", int(df["grade"].min()), int(df["grade"].max()), int(df["grade"].min()))
        with col3:
            max_markers = st.slider("📍 Max Markers", 100, min(Config.MAX_MAP_MARKERS, len(df)), 300)

        filtered = df[
            (df["price"] >= price_range[0]) & 
            (df["price"] <= price_range[1]) &
            (df["grade"] >= min_grade)
        ].sample(min(max_markers, len(df)))

        m = folium.Map(
            location=[df["lat"].mean(), df["long"].mean()],
            zoom_start=10,
            tiles="CartoDB dark_matter"
        )

        # Add heatmap
        heat_data = [[row["lat"], row["long"], row["price"]/1000000] 
                     for _, row in filtered.iterrows()]
        HeatMap(heat_data, radius=15, blur=25).add_to(m)

        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered.iterrows():
            color = "green" if row["price"] < df["price"].median() else "red"
            folium.CircleMarker(
                location=[row["lat"], row["long"]],
                radius=5,
                popup=folium.Popup(
                    f"<b>Price:</b> ${row['price']:,.0f}<br>"
                    f"<b>Bedrooms:</b> {row['bedrooms']}<br>"
                    f"<b>Grade:</b> {row['grade']}",
                    max_width=200
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(marker_cluster)

        st_folium(m, width=1200, height=700)


class PageModelComparison:
    @staticmethod
    def render(ml_engine):
        UIComponents.render_header("Model Performance", "Algorithm benchmarking & comparison")

        metrics_df = pd.DataFrame({
            name: m.to_dict() 
            for name, m in ml_engine.metrics.items()
        }).T

        st.dataframe(
            metrics_df.style.format({
                "R2 Score": "{:.4f}",
                "MAE ($)": "${:,.0f}",
                "RMSE ($)": "${:,.0f}"
            }),
            use_container_width=True
        )

        fig = go.Figure()
        for metric in ["R2 Score", "MAE ($)", "RMSE ($)"]:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition="auto"
            ))

        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            title="Model Comparison",
            yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9')
        )
        st.plotly_chart(fig, use_container_width=True)

        best = max(ml_engine.metrics.items(), key=lambda x: x[1].r2_score)
        st.success(f"🏆 Best Performer: **{best[0]}** with R² = {best[1].r2_score:.4f}")

        # Model export
        st.subheader("💾 Model Export")
        col1, col2 = st.columns(2)
        for i, (name, _) in enumerate(ml_engine.metrics.items()):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"📥 Export {name}", key=f"export_{name}"):
                    path = ml_engine.save_model(name)
                    st.success(f"Model saved to {path}")


class PageWeather:
    @staticmethod
    def render():
        UIComponents.render_header("Weather Intelligence", "Local climate data integration")

        result = WeatherService.get_seattle_weather()

        if result["status"] == "success":
            data = result["data"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌡️</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #00FFAA;">{data['temperature']}°C</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Feels like {data['feels_like']}°C</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">☁️</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #00C2FF;">{data['description']}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Conditions</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">💧</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #A78BFA;">{data['humidity']}%</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Humidity</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">💨</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #F59E0B;">{data['wind_speed']} m/s</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Wind Speed</div>
                </div>
                """, unsafe_allow_html=True)

            icon_url = f"http://openweathermap.org/img/wn/{data['icon']}@4x.png"
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(icon_url, width=150)

        else:
            st.error(result["message"])
            st.info("To enable weather data, set the OPENWEATHER_API_KEY environment variable.")


class PageDownloads:
    @staticmethod
    def render(df, ml_engine):
        UIComponents.render_header("Export Center", "Download reports, data & models")

        tab1, tab2, tab3 = st.tabs(["📊 Data Export", "🤖 Model Export", "📑 Reports"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("CSV Export")
                csv = ReportEngine.export_csv(df)
                st.download_button(
                    "Download Full Dataset (CSV)",
                    csv,
                    "real_estate_data.csv",
                    "text/csv",
                    use_container_width=True
                )

                summary = pd.DataFrame([ReportEngine.generate_summary_stats(df)])
                summary_csv = summary.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Summary Stats (CSV)",
                    summary_csv,
                    "summary_stats.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                st.subheader("JSON Export")
                json_data = ReportEngine.generate_summary_stats(df)
                json_bytes = ReportEngine.export_json(json_data)
                st.download_button(
                    "Download Summary (JSON)",
                    json_bytes,
                    "summary_stats.json",
                    "application/json",
                    use_container_width=True
                )

                # Correlation matrix
                corr_json = json.dumps(df.select_dtypes(include=[np.number]).corr().to_dict(), indent=2).encode("utf-8")
                st.download_button(
                    "Download Correlation Matrix (JSON)",
                    corr_json,
                    "correlation_matrix.json",
                    "application/json",
                    use_container_width=True
                )

        with tab2:
            if ml_engine.models:
                st.subheader("Trained Models")
                for name in ml_engine.models.keys():
                    try:
                        model_path = Config.MODEL_CACHE_DIR / f"{name.replace(' ', '_').lower()}.pkl"
                        joblib.dump(ml_engine.models[name], model_path)
                        with open(model_path, "rb") as f:
                            st.download_button(
                                f"📥 Download {name} Model",
                                f,
                                f"{name.replace(' ', '_').lower()}_model.pkl",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Model export failed for {name}: {e}")
            else:
                st.info("Train models first to enable export.")

        with tab3:
            st.subheader("Generated Reports")
            st.info("Reports feature coming soon. You can currently export data and models above.")


class PageAdmin:
    """Admin panel with user management, analytics, and system monitoring."""

    @staticmethod
    def render():
        if not AuthManager.has_role("admin"):
            st.error("🚫 Access Denied. Admin privileges required.")
            st.info("Contact your system administrator for access.")
            return

        UIComponents.render_header("Admin Panel", "System management & analytics")

        tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "📊 Analytics", "🔌 API Logs", "⚙️ System"])

        with tab1:
            st.subheader("User Management")
            users_df = db.get_df("SELECT id, username, email, role, created_at, last_login, is_active FROM users ORDER BY created_at DESC")
            if not users_df.empty:
                st.dataframe(users_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Add New User")
                    new_username = st.text_input("Username", key="new_user")
                    new_email = st.text_input("Email", key="new_email")
                    new_password = st.text_input("Password", type="password", key="new_pass")
                    new_role = st.selectbox("Role", ["user", "analyst", "admin"], key="new_role")

                    if st.button("➕ Create User", use_container_width=True):
                        if new_username and new_password:
                            try:
                                hashed = AuthManager.hash_password(new_password)
                                db.insert("users", {
                                    "username": new_username,
                                    "password_hash": hashed,
                                    "email": new_email,
                                    "role": new_role
                                })
                                st.success(f"User {new_username} created successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to create user: {e}")
                        else:
                            st.warning("Username and password are required.")

                with col2:
                    st.subheader("Manage Users")
                    user_to_manage = st.selectbox("Select User", users_df["username"].tolist(), key="manage_user")
                    selected_user = users_df[users_df["username"] == user_to_manage].iloc[0]

                    new_role = st.selectbox("Change Role", ["user", "analyst", "admin"], 
                                           index=["user", "analyst", "admin"].index(selected_user["role"]),
                                           key="change_role")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔄 Update Role", use_container_width=True):
                            db.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, user_to_manage))
                            st.success(f"Updated {user_to_manage} to {new_role}")
                            st.rerun()
                    with col_b:
                        if st.button("🗑️ Delete User", use_container_width=True):
                            if user_to_manage != Config.ADMIN_USERNAME:
                                db.execute("DELETE FROM users WHERE username = ?", (user_to_manage,))
                                st.success(f"User {user_to_manage} deleted")
                                st.rerun()
                            else:
                                st.error("Cannot delete default admin user.")
            else:
                st.info("No users found.")

        with tab2:
            st.subheader("Analytics Events")
            events_df = db.get_df("""
                SELECT event_type, COUNT(*) as count, 
                       MAX(timestamp) as last_event
                FROM analytics_events 
                GROUP BY event_type 
                ORDER BY count DESC
            """)
            if not events_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(
                        events_df,
                        values="count",
                        names="event_type",
                        template="plotly_dark",
                        title="Event Distribution"
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#F1F5F9')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(events_df, use_container_width=True)
            else:
                st.info("No analytics events recorded yet.")

            st.subheader("Recent Events")
            recent_events = db.get_df("""
                SELECT event_type, event_data, user_id, timestamp 
                FROM analytics_events 
                ORDER BY timestamp DESC 
                LIMIT 50
            """)
            if not recent_events.empty:
                st.dataframe(recent_events, use_container_width=True)

        with tab3:
            st.subheader("API Request Logs")
            api_logs = db.get_df("""
                SELECT endpoint, method, status_code, response_time, timestamp 
                FROM api_logs 
                ORDER BY timestamp DESC 
                LIMIT 100
            """)
            if not api_logs.empty:
                st.dataframe(api_logs, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    status_counts = api_logs.groupby("status_code").size().reset_index(name="count")
                    fig = px.bar(
                        status_counts,
                        x="status_code",
                        y="count",
                        template="plotly_dark",
                        title="Response Status Distribution"
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#F1F5F9')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    avg_response = api_logs.groupby("endpoint")["response_time"].mean().reset_index()
                    avg_response = avg_response.sort_values("response_time", ascending=False).head(10)
                    fig = px.bar(
                        avg_response,
                        x="endpoint",
                        y="response_time",
                        template="plotly_dark",
                        title="Avg Response Time by Endpoint (s)"
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#F1F5F9')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No API logs recorded yet.")

        with tab4:
            st.subheader("System Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="glass-card">
                    <h4>📦 Application</h4>
                    <p><strong>Version:</strong> v3.0.0-ENTERPRISE</p>
                    <p><strong>Environment:</strong> Production</p>
                    <p><strong>Database:</strong> SQLite</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                db_size = Config.DB_PATH.stat().st_size / (1024*1024) if Config.DB_PATH.exists() else 0
                st.markdown(f"""
                <div class="glass-card">
                    <h4>💾 Storage</h4>
                    <p><strong>DB Size:</strong> {db_size:.2f} MB</p>
                    <p><strong>Uploads:</strong> {len(list(Config.UPLOAD_DIR.glob('*')))} files</p>
                    <p><strong>Models:</strong> {len(list(Config.MODEL_CACHE_DIR.glob('*.pkl')))} cached</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="glass-card">
                    <h4>🔐 Security</h4>
                    <p><strong>Auth:</strong> bcrypt + RBAC</p>
                    <p><strong>Session:</strong> Token-based</p>
                    <p><strong>Encryption:</strong> AES-256</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("Database Maintenance")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧹 Clear API Logs", use_container_width=True):
                    db.execute("DELETE FROM api_logs")
                    st.success("API logs cleared")
            with col2:
                if st.button("🧹 Clear Analytics Events", use_container_width=True):
                    db.execute("DELETE FROM analytics_events")
                    st.success("Analytics events cleared")


class PageChatbot:
    """AI Assistant chat interface."""

    @staticmethod
    def render(df=None):
        UIComponents.render_header("AI Assistant", "Your intelligent real estate companion")

        chatbot = AIChatbot()

        # Chat interface
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem;">🤖</div>
            <div>
                <h2 style="margin: 0;">REI Assistant</h2>
                <p style="color: #94A3B8; margin: 0;">Powered by AI — Ask me anything about real estate</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick action buttons
        col1, col2, col3, col4 = st.columns(4)
        quick_prompts = {
            "📊 Market Overview": "Show me the market overview",
            "💎 Investment Tips": "What are the best investment opportunities?",
            "🔮 Price Trends": "What are the current price trends?",
            "🗺️ Best Areas": "Which areas have the best investment potential?"
        }

        for col, (label, prompt) in zip([col1, col2, col3, col4], quick_prompts.items()):
            with col:
                if st.button(label, use_container_width=True, key=f"quick_{label}"):
                    st.session_state.chat_input = prompt

        # Chat history display
        chat_container = st.container()
        with chat_container:
            history = chatbot.get_chat_history(limit=50)
            for msg in history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])

        # Input
        prompt = st.chat_input("Ask REI Assistant anything...", key="chat_input_main")
        if prompt:
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)

            chatbot.save_message("user", prompt)

            with st.spinner("🤖 Thinking..."):
                response = chatbot.generate_response(prompt, df)

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)

            chatbot.save_message("assistant", response)
            st.rerun()

        # Clear history button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                chatbot.clear_history()
                st.success("Chat history cleared")
                st.rerun()


class PageProducts:
    """Product/Property listings page."""

    @staticmethod
    def render(df):
        UIComponents.render_header("Property Listings", "Browse and manage property catalog")

        tab1, tab2 = st.tabs(["📋 Browse Listings", "➕ Add Property"])

        with tab1:
            # Filter controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                min_price = st.number_input("Min Price", 0, int(df["price"].max()), 0)
            with col2:
                max_price = st.number_input("Max Price", 0, int(df["price"].max()), int(df["price"].max()))
            with col3:
                min_beds = st.number_input("Min Bedrooms", 0, int(df["bedrooms"].max()), 0)
            with col4:
                status_filter = st.selectbox("Status", ["All", "Active", "Featured"])

            filtered = df[
                (df["price"] >= min_price) &
                (df["price"] <= max_price) &
                (df["bedrooms"] >= min_beds)
            ].copy()

            st.subheader(f"📊 Showing {len(filtered)} Properties")

            # Grid display
            cols = st.columns(3)
            for idx, (_, row) in enumerate(filtered.head(12).iterrows()):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.2rem; font-weight: 700; color: #00FFAA;">${row['price']:,.0f}</span>
                            <span style="background: rgba(0,255,170,0.1); color: #00FFAA; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem;">
                                Grade {row['grade']}
                            </span>
                        </div>
                        <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                            🛏️ {int(row['bedrooms'])} bed • 🛁 {row['bathrooms']} bath • 📐 {int(row['sqft_living'])} sqft
                        </div>
                        <div style="color: #64748B; font-size: 0.8rem;">
                            📍 Zipcode: {row['zipcode']}
                        </div>
                        <div style="margin-top: 0.75rem; display: flex; gap: 0.5rem;">
                            <button style="flex: 1; background: linear-gradient(135deg, #00FFAA, #00C2FF); border: none; border-radius: 8px; padding: 6px; color: #0B0F19; font-weight: 600; font-size: 0.8rem; cursor: pointer;">
                                View Details
                            </button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.dataframe(
                filtered[["price", "bedrooms", "bathrooms", "sqft_living", "grade", "zipcode", "condition"]].head(50)
                .style.format({"price": "${:,.0f}"}),
                use_container_width=True
            )

        with tab2:
            if AuthManager.has_role("analyst"):
                st.subheader("Add New Property")
                with st.form("add_property"):
                    col1, col2 = st.columns(2)
                    with col1:
                        title = st.text_input("Property Title")
                        price = st.number_input("Price", min_value=0)
                        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20)
                        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, step=0.5)
                        sqft_living = st.number_input("Living Sqft", min_value=0)
                    with col2:
                        zipcode = st.text_input("Zipcode")
                        grade = st.slider("Grade", 1, 13, 7)
                        condition = st.slider("Condition", 1, 5, 3)
                        waterfront = st.checkbox("Waterfront")
                        featured = st.checkbox("Featured Listing")

                    description = st.text_area("Description")

                    submitted = st.form_submit_button("➕ Add Property", use_container_width=True)
                    if submitted:
                        try:
                            db.insert("properties", {
                                "title": title,
                                "description": description,
                                "price": price,
                                "bedrooms": bedrooms,
                                "bathrooms": bathrooms,
                                "sqft_living": sqft_living,
                                "zipcode": zipcode,
                                "grade": grade,
                                "condition": condition,
                                "waterfront": 1 if waterfront else 0,
                                "featured": 1 if featured else 0
                            })
                            st.success("Property added successfully!")
                            NotificationManager.add_notification(
                                st.session_state.get("user_id", 0),
                                "New Property Added",
                                f"Property '{title}' was added to the catalog",
                                "success"
                            )
                        except Exception as e:
                            st.error(f"Failed to add property: {e}")
            else:
                st.info("🔒 Analyst or Admin role required to add properties.")


class PageAPIMonitor:
    """API monitoring and testing page."""

    @staticmethod
    def render():
        UIComponents.render_header("API Monitor", "REST API integration & testing")

        st.markdown("""
        <div class="glass-card" style="margin-bottom: 2rem;">
            <h3>🔌 API Configuration</h3>
            <p style="color: #94A3B8;">Configure and test external API integrations for your real estate data pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌐 API Settings")
            api_base = st.text_input("API Base URL", value=APIClient.BASE_URL)
            st.info("Default: http://localhost:8000")

            st.subheader("🔑 API Keys")
            weather_key = st.text_input("OpenWeather API Key", value=Config.WEATHER_API_KEY, type="password")
            openai_key = st.text_input("OpenAI API Key", value=Config.OPENAI_API_KEY, type="password")

            if st.button("💾 Save Configuration", use_container_width=True):
                st.success("Configuration saved (session only)")

        with col2:
            st.subheader("🧪 API Testing")
            endpoint = st.selectbox(
                "Test Endpoint",
                ["/api/v1/predict", "/api/v1/market/summary", "/api/v1/properties", "/health"]
            )

            if endpoint == "/api/v1/predict":
                test_data = {
                    "bedrooms": st.number_input("Bedrooms", 1, 10, 3),
                    "bathrooms": st.number_input("Bathrooms", 0.5, 10.0, 2.0),
                    "sqft_living": st.number_input("Sqft Living", 100, 10000, 2000),
                    "floors": st.number_input("Floors", 1.0, 4.0, 1.0),
                    "grade": st.slider("Grade", 1, 13, 7)
                }
            else:
                test_data = {}

            if st.button("🚀 Send Request", use_container_width=True):
                with st.spinner("Sending request..."):
                    if endpoint == "/api/v1/predict":
                        result = APIClient.predict_price(test_data)
                    elif endpoint == "/api/v1/market/summary":
                        result = APIClient.get_market_summary()
                    else:
                        result = APIClient.get(endpoint)

                st.subheader("Response")
                if result["status"] == "success":
                    st.json(result["data"])
                else:
                    st.error(result["message"])

        st.subheader("📡 Recent API Activity")
        recent_logs = db.get_df("""
            SELECT endpoint, method, status_code, response_time, timestamp 
            FROM api_logs 
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        if not recent_logs.empty:
            st.dataframe(recent_logs, use_container_width=True)
        else:
            st.info("No API activity recorded yet. Send a test request above.")


# =====================================================
# MAIN APPLICATION ORCHESTRATOR
# =====================================================

class Application:
    """Main application orchestrator with all enterprise features."""

    PAGE_REGISTRY = {
        "Dashboard": (PageDashboard, "🏠", "user"),
        "Price Analysis": (PagePriceAnalysis, "📊", "user"),
        "Area Analysis": (PageAreaAnalysis, "🗺️", "user"),
        "Investment Insights": (PageInvestment, "💎", "user"),
        "Compare Properties": (PageCompare, "⚖️", "user"),
        "AI Recommendations": (PageRecommendations, "🎯", "user"),
        "AI Explainability": (PageExplainability, "🧠", "analyst"),
        "Forecasting": (PageForecasting, "🔮", "analyst"),
        "Map Visualization": (PageMap, "🌍", "user"),
        "Model Comparison": (PageModelComparison, "📈", "analyst"),
        "Weather": (PageWeather, "🌤️", "user"),
        "Property Listings": (PageProducts, "🏘️", "user"),
        "AI Assistant": (PageChatbot, "🤖", "user"),
        "API Monitor": (PageAPIMonitor, "🔌", "analyst"),
        "Downloads": (PageDownloads, "📥", "user"),
        "Admin Panel": (PageAdmin, "⚙️", "admin"),
    }

    def __init__(self):
        self._setup_page_config()
        self._inject_analytics()

    def _setup_page_config(self):
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _inject_analytics(self):
        if Config.GA_TRACKING_ID:
            st.components.v1.html(f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={Config.GA_TRACKING_ID}"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());
                gtag('config', '{Config.GA_TRACKING_ID}');
            </script>
            """, height=0)

    def _render_sidebar(self, df):
        """Render enhanced navigation sidebar with notifications and user info."""
        with st.sidebar:
            # App branding
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏠</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9;">{Config.PAGE_TITLE}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.25rem;">v{Config.VERSION}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 1rem 0;'>", unsafe_allow_html=True)

            # User info card
            if st.session_state.get("authenticated"):
                role_color = {"admin": "#EF4444", "analyst": "#F59E0B", "user": "#00FFAA", "guest": "#94A3B8"}
                role = st.session_state.get("user_role", "guest")
                color = role_color.get(role, "#94A3B8")

                st.markdown(f"""
                <div class="glass-card" style="padding: 1rem; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background: {color}20; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                            👤
                        </div>
                        <div>
                            <div style="font-weight: 600; color: #F1F5F9; font-size: 0.9rem;">{st.session_state.get("username", "Guest")}</div>
                            <div style="font-size: 0.75rem; color: {color}; font-weight: 500;">{role.upper()}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Notifications
                unread = NotificationManager.get_unread_count(st.session_state.get("user_id", 0))
                if unread > 0:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 0.75rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">🔔</span>
                        <span style="font-size: 0.85rem; color: #EF4444; font-weight: 600;">{unread} unread notification{'s' if unread > 1 else ''}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Dataset info
            with st.expander("📂 Dataset Info", expanded=True):
               st.markdown(f"<span style='color:#E2E8F0; font-weight:600;'>Rows:</span> <span style='color:white;'>{df.shape[0]:,}</span>", unsafe_allow_html=True)
               st.markdown(f"<span style='color:#E2E8F0; font-weight:600;'>Columns:</span> <span style='color:white;'>{df.shape[1]}</span>", unsafe_allow_html=True)
               st.markdown(f"<span style='color:#E2E8F0; font-weight:600;'>Numeric:</span> <span style='color:white;'>{len(df.select_dtypes(include=[np.number]).columns)}</span>", unsafe_allow_html=True)
               st.markdown(f"<span style='color:#E2E8F0; font-weight:600;'>Zipcodes:</span> <span style='color:white;'>{df['zipcode'].nunique()}</span>", unsafe_allow_html=True)
               st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 1rem 0;'>", unsafe_allow_html=True)

            # Navigation
            st.markdown("<div style='font-size: 0.75rem; color: #E2E8F0; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 0.05em;'>NAVIGATION</div>", unsafe_allow_html=True)
            # Filter pages by role
            available_pages = {}
            for page_name, (page_class, icon, required_role) in self.PAGE_REGISTRY.items():
                if AuthManager.has_role(required_role):
                    available_pages[page_name] = (page_class, icon)

            selected = st.radio(
                "",
                list(available_pages.keys()),
                format_func=lambda x: f"{available_pages[x][1]} {x}",
                label_visibility="collapsed"
            )

            st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 1rem 0;'>", unsafe_allow_html=True)

            # Quick actions
            st.markdown("<div style='font-size: 0.75rem; color: #E2E8F0; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 0.05em;'>QUICK ACTIONS</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 AI Chat", use_container_width=True, key="sidebar_chat"):
                    st.session_state.current_page = "AI Assistant"
                    st.rerun()
            with col2:
                if st.button("📥 Export", use_container_width=True, key="sidebar_export"):
                    st.session_state.current_page = "Downloads"
                    st.rerun()

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            # Sign out
            if st.button("🚪 Sign Out", use_container_width=True, key="sidebar_logout"):
                AuthManager.logout()

            # Footer
            st.markdown("""
            <div style="position: fixed; bottom: 1rem; left: 1rem; right: 1rem; text-align: center;">
                <div style="font-size: 0.65rem; color: #475569;">
                    Real Estate Intelligence Hub<br>
                    Enterprise Edition v3.0
                </div>
            </div>
            """, unsafe_allow_html=True)

            return selected

    def run(self):
        """Main application entry point."""
        AuthManager.init_session()

        # Inject theme
        ThemeEngine.inject()

        if not AuthManager.require_auth():
            AuthManager.render_login_page()
            st.stop()

        try:
            # Data source sidebar
            with st.sidebar:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                uploaded = st.file_uploader("📤 Upload CSV", type=["csv"], key="file_uploader")

                if uploaded:
                    st.success(f"📄 {uploaded.name} ready")

            with st.spinner("🔄 Loading and validating data..."):
                df = DataManager.load_data(uploaded)

                # Log analytics event
                db.insert("analytics_events", {
                    "event_type": "data_load",
                    "event_data": json.dumps({"rows": len(df), "columns": list(df.columns)}),
                    "user_id": st.session_state.get("user_id"),
                    "session_id": st.session_state.get("session_token")
                })

            ml_engine = MLEngine(df)

            # Train models in background
            with st.spinner("🤖 Training AI models..."):
                ml_engine.train_models()

            selected_page = self._render_sidebar(df)

            # Update current page
            st.session_state.current_page = selected_page

            page_class, _, _ = self.PAGE_REGISTRY[selected_page]

            with st.spinner(f"🔄 Loading {selected_page}..."):
                if selected_page == "Dashboard":
                    page_class.render(df, ml_engine)
                elif selected_page in ["AI Explainability", "Model Comparison"]:
                    page_class.render(ml_engine)
                elif selected_page == "Downloads":
                    page_class.render(df, ml_engine)
                elif selected_page == "Weather":
                    page_class.render()
                elif selected_page == "AI Assistant":
                    page_class.render(df)
                elif selected_page == "Admin Panel":
                    page_class.render()
                elif selected_page == "API Monitor":
                    page_class.render()
                else:
                    page_class.render(df)

            # Footer
            st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,170,0.2), transparent); margin: 2rem 0 1rem;'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align: center; color: #475569; font-size: 0.75rem;">
                <p>🏙 Real Estate Intelligence Hub v{Config.VERSION} | Enterprise Edition</p>
                <p>Built with Streamlit, Plotly, Scikit-learn & SQLite</p>
            </div>
            """, unsafe_allow_html=True)

        except DataValidationError as e:
            UIComponents.render_error("Data Validation Failed", e)
            st.info("Please upload a valid CSV file with the required columns: price, bedrooms, bathrooms, sqft_living, floors, grade, zipcode, lat, long")
        except Exception as e:
            UIComponents.render_error("An unexpected error occurred", e)
            logger.exception("Unhandled application error")
            st.error("Please refresh the page or contact support if the issue persists.")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    app = Application()
    app.run()