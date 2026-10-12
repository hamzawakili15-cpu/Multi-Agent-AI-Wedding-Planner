from __future__ import annotations

import os
import textwrap
import time
from datetime import date, datetime, timedelta
from html import escape
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage

from prompts import USER_PROMPT_FOR_MAIN_AGENT, WEDDING_PLANNER_AGENT_PROMPT


APP_TITLE = "Multi-Agent Wedding Planner"
MODEL_NAME = "gpt-5-nano"

API_KEY_CONFIGS = {
    "OPENAI_API_KEY": {"label": "OpenAI", "required": False},
    "ANTHROPIC_API_KEY": {"label": "Anthropic", "required": False},
    "GROQ_API_KEY": {"label": "Groq", "required": False},
    "OMNIROUTE_API_KEY": {"label": "OmniRoute", "required": False},
    "TAVILY_API_KEY": {"label": "Tavily (search)", "required": True},
}

LLM_PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "OMNIROUTE_API_KEY")

STYLE_OPTIONS = [
    "Modern",
    "Romantic",
    "Classic",
    "Garden",
    "Minimal",
    "Luxury",
    "Cultural fusion",
    "Destination",
    "Eco-conscious",
    "Black tie",
]

PRIORITY_OPTIONS = [
    "Venue shortlist",
    "Vendor research",
    "Budget allocation",
    "Guest experience",
    "Timeline",
    "Design direction",
    "Risk management",
    "Travel logistics",
]


load_dotenv()
st.set_page_config(page_title=APP_TITLE, page_icon="💍", layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');

        :root {
            --ink: #0f172a;
            --ink-soft: #1e293b;
            --muted: #64748b;
            --line: rgba(148, 163, 184, 0.2);
            --line-strong: rgba(148, 163, 184, 0.35);
            --panel: rgba(255, 255, 255, 0.82);
            --panel-solid: #ffffff;
            --surface: #f8fafc;
            --surface-strong: #f1f5f9;
            --sidebar-bg: linear-gradient(175deg, #0f172a 0%, #1a1f3a 48%, #1e1b2e 100%);
            --sidebar-field: rgba(255, 255, 255, 0.06);
            --accent: #7c3aed;
            --accent-light: rgba(124, 58, 237, 0.08);
            --accent-mid: rgba(124, 58, 237, 0.15);
            --coral: #e11d48;
            --coral-soft: rgba(225, 29, 72, 0.08);
            --gold: #d97706;
            --gold-soft: rgba(217, 119, 6, 0.1);
            --emerald: #059669;
            --emerald-soft: rgba(5, 150, 105, 0.1);
            --rose: #f43f5e;
            --rose-soft: rgba(244, 63, 94, 0.08);
            --sky: #0ea5e9;
            --sky-soft: rgba(14, 165, 233, 0.08);
            --glass: rgba(255, 255, 255, 0.6);
            --glass-border: rgba(255, 255, 255, 0.5);
            --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
            --shadow: 0 4px 24px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04);
            --shadow-lg: 0 12px 48px rgba(15, 23, 42, 0.08), 0 4px 16px rgba(15, 23, 42, 0.04);
            --shadow-glow: 0 0 40px rgba(124, 58, 237, 0.08);
            --radius: 14px;
            --radius-sm: 10px;
            --radius-pill: 999px;
            --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-bounce: 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(22px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes scaleIn {
            from { opacity: 0; transform: scale(0.92); }
            to   { opacity: 1; transform: scale(1); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-18px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes pulse-ring {
            0%   { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.3); }
            70%  { box-shadow: 0 0 0 8px rgba(124, 58, 237, 0); }
            100% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-6px); }
        }
        @keyframes gradient-shift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse 80% 60% at 20% -10%, rgba(124, 58, 237, 0.06), transparent),
                radial-gradient(ellipse 60% 50% at 80% 10%, rgba(225, 29, 72, 0.04), transparent),
                radial-gradient(ellipse 50% 40% at 50% 90%, rgba(14, 165, 233, 0.03), transparent),
                linear-gradient(180deg, #ffffff 0%, var(--surface) 30%, var(--surface) 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        .block-container {
            max-width: 1280px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p, div, span, label { letter-spacing: -0.01em; }

        /* ---- Header ---- */
        .app-header {
            animation: fadeInUp 0.7s ease-out both;
            background: var(--panel);
            backdrop-filter: blur(20px) saturate(1.4);
            -webkit-backdrop-filter: blur(20px) saturate(1.4);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow-lg), var(--shadow-glow);
            display: grid;
            gap: 1.5rem;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: end;
            padding: 1.6rem 1.8rem 1.5rem;
            margin-bottom: 1.3rem;
            position: relative;
            overflow: hidden;
        }
        .app-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), var(--coral), var(--gold), var(--accent));
            background-size: 300% 100%;
            animation: gradient-shift 6s ease infinite;
        }
        .app-header::after {
            content: '';
            position: absolute;
            top: -50%; right: -20%;
            width: 400px; height: 400px;
            background: radial-gradient(circle, rgba(124, 58, 237, 0.04), transparent 70%);
            pointer-events: none;
        }

        .app-kicker {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }
        .app-kicker::before {
            content: '';
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--accent);
            animation: pulse-ring 2s ease infinite;
        }

        .app-title {
            font-family: 'Playfair Display', serif;
            color: var(--ink);
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0 0 0.5rem;
            background: linear-gradient(135deg, var(--ink) 0%, #334155 50%, var(--ink) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.6;
            max-width: 600px;
            margin: 0;
            font-weight: 400;
        }

        .header-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: flex-end;
            max-width: 320px;
            animation: fadeIn 0.9s ease-out 0.3s both;
        }

        .header-chip {
            background: var(--accent-light);
            border: 1px solid var(--accent-mid);
            border-radius: var(--radius-pill);
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 650;
            padding: 0.3rem 0.75rem;
            white-space: nowrap;
            transition: all var(--transition);
            cursor: default;
        }
        .header-chip:hover {
            background: var(--accent-mid);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.12);
        }

        /* ---- Summary Grid ---- */
        .summary-grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0 0 1.2rem;
        }

        .summary-card {
            animation: scaleIn 0.5s ease-out both;
            background: var(--panel);
            backdrop-filter: blur(16px);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            min-height: 6rem;
            padding: 1rem 1.1rem;
            position: relative;
            overflow: hidden;
            transition: all var(--transition);
        }
        .summary-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
            border-color: rgba(124, 58, 237, 0.2);
        }
        .summary-card:nth-child(1) { animation-delay: 0.1s; }
        .summary-card:nth-child(2) { animation-delay: 0.18s; }
        .summary-card:nth-child(3) { animation-delay: 0.26s; }
        .summary-card:nth-child(4) { animation-delay: 0.34s; }

        .summary-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 3px;
            border-radius: var(--radius) var(--radius) 0 0;
        }
        .summary-card:nth-child(1)::before { background: linear-gradient(90deg, var(--emerald), #34d399); }
        .summary-card:nth-child(2)::before { background: linear-gradient(90deg, var(--accent), #a78bfa); }
        .summary-card:nth-child(3)::before { background: linear-gradient(90deg, var(--gold), #fbbf24); }
        .summary-card:nth-child(4)::before { background: linear-gradient(90deg, var(--coral), #fb7185); }

        .summary-card .icon-badge {
            width: 32px; height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }
        .summary-card:nth-child(1) .icon-badge { background: var(--emerald-soft); }
        .summary-card:nth-child(2) .icon-badge { background: var(--accent-light); }
        .summary-card:nth-child(3) .icon-badge { background: var(--gold-soft); }
        .summary-card:nth-child(4) .icon-badge { background: var(--rose-soft); }

        .summary-card .label {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .summary-card .value {
            color: var(--ink);
            display: block;
            font-size: 1.3rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.15rem;
        }

        .summary-card .note {
            color: var(--muted);
            display: block;
            font-size: 0.8rem;
            line-height: 1.3;
            font-weight: 450;
        }

        /* ---- Section Labels ---- */
        .section-label {
            color: var(--ink);
            font-weight: 800;
            font-size: 1.05rem;
            margin: 0.4rem 0 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            animation: fadeIn 0.5s ease-out both;
        }
        .section-label::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--line-strong), transparent);
        }

        .field-group-title {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            margin: 0.3rem 0 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .field-group-title::before {
            content: '';
            width: 3px; height: 14px;
            background: var(--accent);
            border-radius: 3px;
        }

        /* ---- Form ---- */
        [data-testid="stForm"] {
            animation: fadeInUp 0.6s ease-out 0.15s both;
            background: var(--panel);
            backdrop-filter: blur(20px);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 1.2rem 1.3rem 1.3rem;
            transition: box-shadow var(--transition);
        }
        [data-testid="stForm"]:hover {
            box-shadow: var(--shadow-lg);
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stForm"] label,
        [data-testid="stForm"] p {
            color: var(--ink-soft) !important;
            font-weight: 600;
            font-size: 0.88rem;
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTextArea textarea {
            background: var(--surface) !important;
            border: 1.5px solid var(--line) !important;
            border-radius: var(--radius-sm) !important;
            box-shadow: none !important;
            color: var(--ink) !important;
            font-weight: 500;
            transition: all var(--transition) !important;
        }
        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stDateInput input:focus,
        .stTextArea textarea:focus {
            background: #ffffff !important;
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1), var(--shadow-sm) !important;
        }

        div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--ink) !important;
            min-height: 2.7rem;
            transition: all var(--transition) !important;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: var(--accent) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        div[data-baseweb="tag"] {
            background: var(--accent-light) !important;
            border: 1px solid var(--accent-mid) !important;
            border-radius: 8px !important;
            transition: all var(--transition) !important;
        }
        div[data-baseweb="tag"] span {
            color: var(--accent) !important;
            font-weight: 650 !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
            border: 3px solid #ffffff !important;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3) !important;
            transition: transform var(--transition-bounce) !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"]:hover {
            transform: scale(1.15) !important;
        }
        .stSlider [data-baseweb="slider"] > div {
            color: var(--accent) !important;
        }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(circle at 30% 20%, rgba(124, 58, 237, 0.1), transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(225, 29, 72, 0.06), transparent 50%);
            pointer-events: none;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: rgba(255, 255, 255, 0.92) !important;
        }

        [data-testid="stSidebar"] .stTextInput input {
            background: var(--sidebar-field) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            border-radius: var(--radius-sm) !important;
            backdrop-filter: blur(8px);
        }
        [data-testid="stSidebar"] .stTextInput input:focus {
            border-color: rgba(124, 58, 237, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.08);
            margin: 1rem 0;
        }

        .sidebar-section-title {
            color: rgba(255, 255, 255, 0.5) !important;
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.6rem;
        }

        /* ---- Side Panels ---- */
        .side-panel {
            animation: slideInLeft 0.6s ease-out both;
            background: var(--panel);
            backdrop-filter: blur(16px);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
            transition: all var(--transition);
        }
        .side-panel:nth-child(2) { animation-delay: 0.12s; }
        .side-panel:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }

        .side-panel h3 {
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 750;
            margin: 0 0 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .side-panel h3::before {
            content: '';
            width: 4px; height: 16px;
            background: linear-gradient(180deg, var(--accent), var(--coral));
            border-radius: 4px;
        }

        .side-panel p {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
            margin: 0 0 0.75rem;
        }

        .lens-grid {
            display: grid;
            gap: 0.6rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-bottom: 0.8rem;
        }

        .lens-item {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: var(--radius-sm);
            padding: 0.65rem 0.7rem;
            transition: all var(--transition);
        }
        .lens-item:hover {
            background: #ffffff;
            border-color: rgba(124, 58, 237, 0.2);
            box-shadow: var(--shadow-sm);
        }

        .lens-item span {
            color: var(--muted);
            display: block;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .lens-item strong {
            color: var(--ink);
            display: block;
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.25;
            margin-top: 0.15rem;
        }

        .chip-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        .soft-chip {
            background: var(--gold-soft);
            border: 1px solid rgba(217, 119, 6, 0.18);
            border-radius: var(--radius-pill);
            color: #92400e;
            font-size: 0.73rem;
            font-weight: 650;
            padding: 0.22rem 0.6rem;
            transition: all var(--transition);
        }
        .soft-chip:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(217, 119, 6, 0.12);
        }

        /* ---- Flow Rows ---- */
        .flow-row {
            border-top: 1px solid var(--line);
            padding: 0.72rem 0;
            transition: all var(--transition);
        }
        .flow-row:first-of-type {
            border-top: 0;
            padding-top: 0;
        }
        .flow-row:hover {
            padding-left: 0.3rem;
        }
        .flow-row strong {
            color: var(--ink);
            display: block;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .flow-row span {
            color: var(--muted);
            display: block;
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .flow-step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px; height: 20px;
            background: var(--accent-light);
            color: var(--accent);
            font-size: 0.65rem;
            font-weight: 800;
            border-radius: 6px;
            margin-right: 0.4rem;
            flex-shrink: 0;
        }

        /* ---- Key Badges ---- */
        .key-row {
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            transition: all var(--transition);
        }
        .key-row:hover {
            padding-left: 0.2rem;
        }
        .key-row span:first-child {
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 0.88rem;
            font-weight: 500;
        }
        .badge {
            border-radius: var(--radius-pill);
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.18rem 0.6rem;
            letter-spacing: 0.02em;
        }
        .badge-ready {
            background: rgba(52, 211, 153, 0.15);
            color: #6ee7b7 !important;
        }
        .badge-missing {
            background: rgba(251, 113, 133, 0.15);
            color: #fda4af !important;
        }

        /* ---- Metrics ---- */
        div[data-testid="stMetric"] {
            animation: scaleIn 0.4s ease-out both;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 0.9rem 1rem;
            transition: all var(--transition);
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }
        div[data-testid="stMetric"] label { color: var(--muted); }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p {
            color: var(--ink) !important;
            font-weight: 650;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div {
            color: var(--ink) !important;
            font-weight: 800;
        }
        div[data-testid="stMetricDelta"] {
            color: var(--muted) !important;
        }

        /* ---- Results ---- */
        .result-shell {
            animation: fadeInUp 0.5s ease-out both;
            background: var(--panel);
            backdrop-filter: blur(16px);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 1.2rem 1.3rem;
            margin-top: 0.8rem;
        }

        .result-title-row {
            animation: fadeIn 0.5s ease-out both;
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: space-between;
            margin: 0.4rem 0 0.8rem;
        }
        .result-title-row h3 {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 750;
            margin: 0;
        }
        .result-title-row span {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 550;
            background: var(--surface);
            padding: 0.2rem 0.6rem;
            border-radius: var(--radius-pill);
        }

        .empty-result {
            animation: fadeIn 0.6s ease-out both;
            background: var(--panel);
            border: 1.5px dashed var(--line-strong);
            border-radius: var(--radius);
            color: var(--muted);
            padding: 2.5rem 1.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .empty-result::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(circle at 50% 50%, rgba(124, 58, 237, 0.03), transparent 60%);
            pointer-events: none;
        }
        .empty-result .empty-icon {
            font-size: 2.2rem;
            margin-bottom: 0.8rem;
            animation: float 3s ease-in-out infinite;
            display: block;
        }
        .empty-result strong {
            color: var(--ink);
            display: block;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .empty-result p {
            color: var(--muted);
            font-size: 0.88rem;
            margin: 0;
            max-width: 380px;
            margin-left: auto;
            margin-right: auto;
        }

        /* ---- Buttons ---- */
        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            background: #ffffff;
            border: 1.5px solid var(--line-strong);
            border-radius: var(--radius-sm);
            color: var(--ink);
            font-weight: 650;
            min-height: 2.8rem;
            transition: all var(--transition);
            position: relative;
            overflow: hidden;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }
        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: translateY(0);
        }

        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, var(--accent) 0%, #6d28d9 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3) !important;
            letter-spacing: 0.01em;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
        }
        div[data-testid="stFormSubmitButton"] button:active {
            transform: translateY(0) !important;
        }
        div[data-testid="stFormSubmitButton"] button:disabled {
            background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%) !important;
            box-shadow: none !important;
            color: rgba(255, 255, 255, 0.6) !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.05);
            border: 1.5px solid rgba(255, 255, 255, 0.12);
            color: rgba(255, 255, 255, 0.85);
            border-radius: var(--radius-sm);
            backdrop-filter: blur(8px);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(124, 58, 237, 0.4);
            color: #ffffff;
            transform: translateY(-1px);
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 2px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all var(--transition);
        }
        .stTabs [aria-selected="true"] {
            border-bottom-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        /* ---- Status / Toast ---- */
        div[data-testid="stStatusWidget"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--line) !important;
            box-shadow: var(--shadow) !important;
        }

        /* ---- Responsive ---- */
        @media (max-width: 768px) {
            .block-container { padding-top: 0.6rem; }
            .app-header {
                grid-template-columns: 1fr;
                padding: 1.1rem;
            }
            .app-title { font-size: 1.8rem; }
            .header-chips {
                justify-content: flex-start;
                max-width: none;
            }
            .summary-grid, .lens-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_default_form_values() -> dict[str, Any]:
    return {
        "couple_name": "Amira and Noah",
        "wedding_location": "Chicago, Illinois",
        "wedding_date": date.today() + timedelta(days=240),
        "guest_count": 120,
        "currency": "USD",
        "budget_range": (35000, 65000),
        "ceremony_type": "Ceremony and reception",
        "wedding_styles": ["Modern", "Romantic", "Cultural fusion"],
        "planning_priorities": ["Venue shortlist", "Vendor research", "Budget allocation", "Timeline"],
        "tone": "Elegant and practical",
        "must_haves": (
            "Indoor-outdoor venue, strong vegetarian menu, live acoustic music, "
            "smooth guest transportation, and a photography-forward schedule."
        ),
        "constraints": (
            "Avoid venues more than 45 minutes from downtown. Keep the design elevated "
            "without relying on heavy floral spend."
        ),
        "traditions": (
            "Blend a short western ceremony with a family tea ceremony. Include space "
            "for speeches from both families."
        ),
    }


def load_sample_values() -> None:
    for key, value in get_default_form_values().items():
        st.session_state[key] = value


def initialize_state() -> None:
    st.session_state.setdefault("plans", [])
    for key, value in get_default_form_values().items():
        st.session_state.setdefault(key, value)


def read_streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def hydrate_env_from_secrets() -> None:
    for key in API_KEY_CONFIGS:
        if os.getenv(key):
            continue
        secret_value = read_streamlit_secret(key)
        if secret_value:
            os.environ[key] = secret_value


def apply_key_overrides(overrides: dict[str, str]) -> None:
    for env_var, value in overrides.items():
        if value.strip():
            os.environ[env_var] = value.strip()


def key_is_ready(name: str) -> bool:
    return bool(os.getenv(name))


def format_budget_range(budget_range: tuple[int, int], currency: str) -> str:
    low, high = budget_range
    return f"{currency} {low // 1000}k - {high // 1000}k"


def format_budget_per_guest(budget_range: tuple[int, int], guest_count: int) -> str:
    if guest_count <= 0:
        return "Not set"
    low, high = budget_range
    average_budget = (low + high) / 2
    return f"{average_budget / guest_count:,.0f} per guest"


def format_days_until(target_date: date) -> str:
    days = (target_date - date.today()).days
    if days == 0:
        return "Today"
    if days == 1:
        return "Tomorrow"
    if days < 0:
        return f"{abs(days)} days ago"
    return f"{days} days out"


def format_list_preview(items: list[str], fallback: str = "Flexible") -> str:
    if not items:
        return fallback
    preview = items[:2]
    suffix = f" +{len(items) - len(preview)}" if len(items) > len(preview) else ""
    return f"{', '.join(preview)}{suffix}"


def render_key_badge(label: str, ready: bool) -> None:
    status_class = "badge-ready" if ready else "badge-missing"
    status_text = "Ready" if ready else "Missing"
    st.markdown(
        f"""
        <div class="key-row">
            <span>{label}</span>
            <span class="badge {status_class}">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="app-kicker">LangChain portfolio project</div>
                <h1 class="app-title">Multi-Agent Wedding Planner</h1>
                <p class="app-subtitle">
                    Build a client-ready planning brief from preferences, constraints,
                    market research, and agent-synthesized recommendations.
                </p>
            </div>
            <div class="header-chips">
                <span class="header-chip">3-agent flow</span>
                <span class="header-chip">Live vendor research</span>
                <span class="header-chip">Markdown export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_summary(runtime_ready: bool) -> None:
    payload = get_form_payload()
    status_value = "Ready" if runtime_ready else "Keys needed"
    status_note = "Planner can run live" if runtime_ready else "Add an LLM + Tavily key"
    guest_note = f"{payload['guest_count']:,} guests"

    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="icon-badge">&#9889;</div>
                <span class="label">Runtime</span>
                <span class="value">{status_value}</span>
                <span class="note">{status_note}</span>
            </div>
            <div class="summary-card">
                <div class="icon-badge">&#128197;</div>
                <span class="label">Event Date</span>
                <span class="value">{format_days_until(payload["wedding_date"])}</span>
                <span class="note">{payload["wedding_date"].strftime("%b %d, %Y")}</span>
            </div>
            <div class="summary-card">
                <div class="icon-badge">&#128176;</div>
                <span class="label">Budget</span>
                <span class="value">{format_budget_range(payload["budget_range"], payload["currency"])}</span>
                <span class="note">{format_budget_per_guest(payload["budget_range"], payload["guest_count"])}</span>
            </div>
            <div class="summary-card">
                <div class="icon-badge">&#127919;</div>
                <span class="label">Planning Focus</span>
                <span class="value">{escape(format_list_preview(payload["planning_priorities"], "Balanced"))}</span>
                <span class="note">{guest_note}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> bool:
    hydrate_env_from_secrets()

    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">LLM Provider Key (any one)</div>', unsafe_allow_html=True)
        overrides: dict[str, str] = {}
        for env_var in LLM_PROVIDER_KEYS:
            cfg = API_KEY_CONFIGS[env_var]
            overrides[env_var] = st.text_input(
                f"{cfg['label']} API key",
                type="password",
                key=f"{env_var}_override",
            )

        st.markdown('<div class="sidebar-section-title">Search Key (required)</div>', unsafe_allow_html=True)
        overrides["TAVILY_API_KEY"] = st.text_input(
            "Tavily API key",
            type="password",
            key="TAVILY_API_KEY_override",
        )

        apply_key_overrides(overrides)

        has_any_llm_key = any(key_is_ready(k) for k in LLM_PROVIDER_KEYS)
        tavily_ready = key_is_ready("TAVILY_API_KEY")

        for env_var, cfg in API_KEY_CONFIGS.items():
            render_key_badge(cfg["label"], key_is_ready(env_var))

        st.divider()
        st.markdown('<div class="sidebar-section-title">Demo Controls</div>', unsafe_allow_html=True)
        st.button("Load sample brief", use_container_width=True, on_click=load_sample_values)

        if st.button("Refresh agent clients", use_container_width=True):
            load_agent_dependencies.clear()
            st.toast("Agent clients refreshed.")

        st.divider()
        st.markdown('<div class="sidebar-section-title">Saved Runs</div>', unsafe_allow_html=True)
        if st.session_state.plans:
            for plan in st.session_state.plans[:4]:
                st.caption(f"{plan['title']} | {plan['created_at']}")
        else:
            st.caption("No runs yet.")

    return has_any_llm_key and tavily_ready


@st.cache_resource(show_spinner=False)
def load_agent_dependencies():
    import importlib

    import agent
    import models

    # Rebuild model + subagents so sidebar key overrides take effect.
    importlib.reload(models)
    importlib.reload(agent)
    from agent import delegate_subagent1, delegate_subagent2
    from models import model

    return model, [delegate_subagent1, delegate_subagent2]


def build_requirements(payload: dict[str, Any]) -> str:
    budget_low, budget_high = payload["budget_range"]
    styles = ", ".join(payload["wedding_styles"]) or "Flexible"
    priorities = ", ".join(payload["planning_priorities"]) or "Balanced planning support"

    return textwrap.dedent(
        f"""
        Couple or event name: {payload["couple_name"]}
        Wedding location: {payload["wedding_location"]}
        Target date: {payload["wedding_date"].strftime("%B %d, %Y")}
        Guest count: {payload["guest_count"]}
        Budget range: {payload["currency"]} {budget_low:,} to {budget_high:,}
        Event scope: {payload["ceremony_type"]}
        Preferred style: {styles}
        Planning priorities: {priorities}
        Planner tone: {payload["tone"]}

        Must-haves:
        {payload["must_haves"].strip() or "None specified."}

        Constraints and sensitivities:
        {payload["constraints"].strip() or "None specified."}

        Cultural, family, or ceremonial details:
        {payload["traditions"].strip() or "None specified."}

        Requested final plan:
        - Executive summary with the planning concept.
        - Venue and vendor research recommendations with clear rationale.
        - Budget allocation by category.
        - Timeline from now through wedding day.
        - Risks, tradeoffs, and open questions for the couple.
        """
    ).strip()


def get_form_payload() -> dict[str, Any]:
    keys = [
        "couple_name",
        "wedding_location",
        "wedding_date",
        "guest_count",
        "currency",
        "budget_range",
        "ceremony_type",
        "wedding_styles",
        "planning_priorities",
        "tone",
        "must_haves",
        "constraints",
        "traditions",
    ]
    return {key: st.session_state[key] for key in keys}


def render_brief_form(runtime_ready: bool) -> bool:
    st.markdown('<div class="section-label">Wedding Brief</div>', unsafe_allow_html=True)

    with st.form("wedding_brief_form"):
        st.markdown('<div class="field-group-title">Basics</div>', unsafe_allow_html=True)
        st.text_input("Couple or project name", key="couple_name")

        col_a, col_b = st.columns(2)
        with col_a:
            st.text_input("Location", key="wedding_location")
            st.number_input("Guest count", min_value=10, max_value=1000, step=10, key="guest_count")
        with col_b:
            st.date_input("Target date", min_value=date.today(), key="wedding_date")
            st.selectbox(
                "Event scope",
                [
                    "Ceremony and reception",
                    "Reception only",
                    "Destination wedding",
                    "Multi-day wedding",
                    "Elopement plus celebration",
                ],
                key="ceremony_type",
            )

        st.markdown('<div class="field-group-title">Budget and Direction</div>', unsafe_allow_html=True)
        budget_col, tone_col = st.columns([1.1, 1])
        with budget_col:
            st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD"], key="currency")
            st.slider(
                "Budget range",
                min_value=5000,
                max_value=250000,
                step=5000,
                key="budget_range",
            )
        with tone_col:
            st.selectbox(
                "Planner tone",
                ["Elegant and practical", "Luxury editorial", "Budget-conscious", "Warm and family-centered"],
                key="tone",
            )
            st.multiselect("Style direction", STYLE_OPTIONS, key="wedding_styles")

        st.markdown('<div class="field-group-title">Priorities and Guardrails</div>', unsafe_allow_html=True)
        st.multiselect("Planning priorities", PRIORITY_OPTIONS, key="planning_priorities")

        notes_a, notes_b = st.columns(2)
        with notes_a:
            st.text_area("Must-haves", height=124, key="must_haves")
            st.text_area("Traditions and family details", height=104, key="traditions")
        with notes_b:
            st.text_area("Constraints", height=244, key="constraints")

        return st.form_submit_button(
            "Generate wedding plan",
            type="primary",
            use_container_width=True,
            disabled=not runtime_ready,
        )


def invoke_planner(requirements: str) -> tuple[str, float]:
    started = time.perf_counter()
    openai_model, delegation_tools = load_agent_dependencies()
    system_prompt = WEDDING_PLANNER_AGENT_PROMPT.format(requirements=requirements)
    main_agent = create_agent(
        model=openai_model,
        tools=delegation_tools,
        name="MainWeddingPlannerAgent",
        system_prompt=system_prompt,
    )
    response = main_agent.invoke({"messages": [HumanMessage(content=USER_PROMPT_FOR_MAIN_AGENT)]})
    content = response["messages"][-1].content
    return content, time.perf_counter() - started


def create_download_name(title: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in title)
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{slug[:44] or 'wedding-plan'}.md"


def save_plan(title: str, requirements: str, content: str, duration: float) -> None:
    st.session_state.plans.insert(
        0,
        {
            "title": title,
            "requirements": requirements,
            "content": content,
            "duration": duration,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )


def render_project_snapshot() -> None:
    st.markdown(
        f"""
        <div class="side-panel">
            <h3>Agent Flow</h3>
            <div class="flow-row">
                <strong><span class="flow-step-num">1</span>MainWeddingPlannerAgent</strong>
                <span>Builds the final strategy from the structured wedding brief.</span>
            </div>
            <div class="flow-row">
                <strong><span class="flow-step-num">2</span>SubAgent1 and SubAgent2</strong>
                <span>Split research tasks for venues, vendors, logistics, and market context.</span>
            </div>
            <div class="flow-row">
                <strong><span class="flow-step-num">3</span>Tavily Search Tool</strong>
                <span>Retrieves current planning data before synthesis.</span>
            </div>
            <div class="flow-row">
                <strong><span class="flow-step-num">4</span>{MODEL_NAME}</strong>
                <span>Coordinates reasoning and client-ready recommendations.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_planning_lens() -> None:
    payload = get_form_payload()
    style_chips = "".join(
        f'<span class="soft-chip">{escape(style)}</span>' for style in payload["wedding_styles"]
    ) or '<span class="soft-chip">Flexible style</span>'
    priority_chips = "".join(
        f'<span class="soft-chip">{escape(priority)}</span>' for priority in payload["planning_priorities"]
    ) or '<span class="soft-chip">Balanced plan</span>'

    st.markdown(
        f"""
        <div class="side-panel">
            <h3>Planning Lens</h3>
            <div class="lens-grid">
                <div class="lens-item">
                    <span>Couple</span>
                    <strong>{escape(payload["couple_name"] or "Untitled")}</strong>
                </div>
                <div class="lens-item">
                    <span>Location</span>
                    <strong>{escape(payload["wedding_location"] or "Not set")}</strong>
                </div>
                <div class="lens-item">
                    <span>Scope</span>
                    <strong>{escape(payload["ceremony_type"])}</strong>
                </div>
                <div class="lens-item">
                    <span>Pace</span>
                    <strong>{format_days_until(payload["wedding_date"])}</strong>
                </div>
            </div>
            <p>Style direction</p>
            <div class="chip-list">{style_chips}</div>
            <p style="margin-top:0.8rem;">Planning priorities</p>
            <div class="chip-list">{priority_chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results() -> None:
    if not st.session_state.plans:
        st.markdown(
            """
            <div class="empty-result">
                <span class="empty-icon">&#128141;</span>
                <strong>No plan generated yet</strong>
                <p>Complete the brief and run the planner to create a client-ready wedding plan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    latest = st.session_state.plans[0]

    st.markdown(
        f"""
        <div class="result-title-row">
            <h3>{escape(latest["title"])}</h3>
            <span>Created {escape(latest["created_at"])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Runtime", f"{latest['duration']:.1f}s")
    metric_b.metric("Brief Words", f"{len(latest['requirements'].split()):,}")
    metric_c.metric("Saved Runs", str(len(st.session_state.plans)))

    plan_tab, requirements_tab, architecture_tab = st.tabs(["Plan", "Input Brief", "Architecture"])

    with plan_tab:
        st.markdown('<div class="result-shell">', unsafe_allow_html=True)
        st.markdown(latest["content"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "Download plan",
            data=latest["content"],
            file_name=create_download_name(latest["title"]),
            mime="text/markdown",
            use_container_width=True,
        )

    with requirements_tab:
        st.code(latest["requirements"], language="markdown")

    with architecture_tab:
        st.markdown(
            """
            ```text
            Streamlit UI
                -> structured wedding requirements
                -> WEDDING_PLANNER_AGENT_PROMPT
                -> MainWeddingPlannerAgent
                    -> delegate_to_subagent1(query)
                        -> SubAgent1 -> Tavily search_web
                    -> delegate_to_subagent2(query)
                        -> SubAgent2 -> Tavily search_web
                -> synthesized wedding plan
            ```
            """
        )


def handle_submission() -> None:
    payload = get_form_payload()
    requirements = build_requirements(payload)
    title = payload["couple_name"].strip() or "Wedding plan"

    with st.status("Running planner agents", expanded=True) as status:
        st.write("Preparing the main planner prompt.")
        st.write("Loading agent dependencies.")
        try:
            st.write("Delegating research and synthesizing the plan.")
            content, duration = invoke_planner(requirements)
        except Exception as exc:
            status.update(label="Planner run failed", state="error")
            st.error("The planner could not complete the run. Check API keys, quotas, and Tavily access.")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

        save_plan(title=title, requirements=requirements, content=content, duration=duration)
        status.update(label="Plan ready", state="complete")


def main() -> None:
    inject_css()
    initialize_state()
    render_header()

    runtime_ready = render_sidebar()

    if not runtime_ready:
        st.warning("Add at least one LLM provider key (OpenAI, Anthropic, Groq, or OmniRoute) and a Tavily key in `.env`, Streamlit secrets, or the sidebar to run the live agents.")

    render_dashboard_summary(runtime_ready=runtime_ready)

    left_col, right_col = st.columns([1.35, 0.8], gap="large")

    with left_col:
        submitted = render_brief_form(runtime_ready=runtime_ready)

    with right_col:
        render_planning_lens()
        render_project_snapshot()

    if submitted:
        handle_submission()

    st.markdown('<div class="section-label">Latest Output</div>', unsafe_allow_html=True)
    render_results()


if __name__ == "__main__":
    main()
