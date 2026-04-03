import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis import STAGE_COLORS, stage_meaning
from data import (
    etf_tickers,
    global_names_ordered,
    kospi_tickers,
    load_all_data,
)

# ─── 페이지 설정 ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="와인스타인 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stage-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .metric-row { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── 사이드바 ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 와인스타인 대시보드")
    st.caption(f"기준일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.divider()

    st.subheader("표시 항목")
    show_global = st.checkbox("🌍 글로벌 자산",  value=True)
    show_etf    = st.checkbox("🏦 ETF",          value=True)
    show_kospi  = st.checkbox("🇰🇷 코스피 종목",  value=True)

    st.divider()

    st.subheader("필터")
    stage_filter = st.multiselect(
        "단계",
        options=list(stage_meaning.keys()),
        default=list(stage_meaning.keys()),
    )
    recently_only = st.checkbox("최근 변경 종목만 표시", value=False)

    st.divider()

    if st.button("🔄 데이터 새로고침", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# ─── 데이터 로딩 ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_data():
    return load_all_data()


progress_placeholder = st.empty()
status_placeholder   = st.empty()

with progress_placeholder:
    progress_bar = st.progress(0, text="데이터 로딩 준비 중...")


def on_progress(idx, total, name):
    pct  = int(idx / total * 100) if total else 100
    text = f"로딩 중 ({idx}/{total}): {name}"
    progress_bar.progress(pct, text=text)


# cache miss 시에만 progress 표시 (cache hit이면 즉시 반환)
results, charts = get_data()

progress_placeholder.empty()
status_placeholder.empty()

if not results:
    st.error("데이터를 불러올 수 없습니다. 네트워크 연결을 확인하세요.")
    st.stop()

result_map   = {r['Asset']: r for r in results}
etf_names    = list(etf_tickers.values())
kospi_names  = list(kospi_tickers.values())

# ─── 헬퍼 함수 ───────────────────────────────────────────────────────────────

def apply_filters(data: list) -> list:
    out = [r for r in data if r['stage'] in stage_filter]
    if recently_only:
        out = [r for r in out if r['recently_changed']]
    return out


def build_table(data: list) -> pd.DataFrame:
    rows = []
    for r in data:
        change_str = (
            r['stage_change_date'].strftime("%Y-%m-%d")
            if r['stage_change_date'] is not None and pd.notna(r['stage_change_date'])
            else "-"
        )
        rows.append({
            "종목":     r['Asset'],
            "현재가":   f"{r['Price']:,}",
            "CN (%)":   r['CN'],
            "단계":     r['stage'] or "-",
            "경과일":   r['elapsed_days'] if r['elapsed_days'] is not None else "-",
            "진입일":   change_str,
            "의미":     stage_meaning.get(r['stage'], "-"),
            "신호":     "🔴 최근변경" if r['recently_changed'] else "",
        })
    return pd.DataFrame(rows)


def style_table(df: pd.DataFrame):
    def color_stage(val):
        c = STAGE_COLORS.get(val, "")
        return f"background-color:{c}22; color:{c}; font-weight:bold" if c else ""

    def color_cn(val):
        try:
            v = float(val)
            if v <= 30:
                return "color:#4CAF50; font-weight:bold"
            if v >= 70:
                return "color:#F44336; font-weight:bold"
        except (TypeError, ValueError):
            pass
        return ""

    return (
        df.style
        .applymap(color_stage, subset=["단계"])
        .applymap(color_cn,    subset=["CN (%)"])
        .format({"CN (%)": "{:.1f}"})
    )


def stage_summary_metrics(data: list):
    cols = st.columns(4)
    for col, stage in zip(cols, stage_meaning):
        count = sum(1 for r in data if r['stage'] == stage)
        color = STAGE_COLORS[stage]
        col.markdown(
            f"<div style='background:{color}22;border-left:4px solid {color};"
            f"padding:8px 12px;border-radius:4px'>"
            f"<div style='color:{color};font-weight:bold;font-size:0.8em'>{stage}</div>"
            f"<div style='font-size:1.6em;font-weight:bold'>{count}</div>"
            f"<div style='font-size:0.75em;color:#888'>{stage_meaning[stage]}</div></div>",
            unsafe_allow_html=True,
        )


def render_stage_tabs(data: list):
    tabs = st.tabs(list(stage_meaning.keys()))
    for tab, stage in zip(tabs, stage_meaning):
        with tab:
            group = [r for r in data if r['stage'] == stage]
            if stage not in stage_filter:
                st.info("현재 필터에서 제외된 단계입니다.")
                continue
            if recently_only:
                group = [r for r in group if r['recently_changed']]
            if not group:
                st.info("해당 종목이 없습니다.")
            else:
                group_sorted = sorted(
                    group,
                    key=lambda x: x['elapsed_days'] if x['elapsed_days'] is not None else 999999,
                )
                st.dataframe(
                    style_table(build_table(group_sorted)),
                    use_container_width=True,
                    hide_index=True,
                )


# ─── 글로벌 추세 ─────────────────────────────────────────────────────────────

if show_global:
    st.header("🌍 글로벌 추세")

    all_global = [result_map[n] for n in global_names_ordered if n in result_map]
    stage_summary_metrics(all_global)
    st.markdown("")

    filtered_global = apply_filters(all_global)
    if filtered_global:
        sorted_global = sorted(
            filtered_global,
            key=lambda x: global_names_ordered.index(x['Asset'])
            if x['Asset'] in global_names_ordered else 999,
        )
        st.dataframe(
            style_table(build_table(sorted_global)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("필터 조건에 맞는 항목이 없습니다.")

# ─── ETF ─────────────────────────────────────────────────────────────────────

if show_etf:
    st.header("🏦 ETF 단계별 분류")
    etf_data = [r for r in results if r['Asset'] in etf_names]
    stage_summary_metrics(etf_data)
    st.markdown("")
    render_stage_tabs(etf_data)

# ─── 코스피 ──────────────────────────────────────────────────────────────────

if show_kospi:
    st.header("🇰🇷 코스피 단계별 분류")
    kospi_data = [r for r in results if r['Asset'] in kospi_names]
    stage_summary_metrics(kospi_data)
    st.markdown("")
    render_stage_tabs(kospi_data)

# ─── 개별 차트 ───────────────────────────────────────────────────────────────

st.header("📊 종목 차트")

chart_names = list(charts.keys())
selected = st.selectbox("종목 선택", options=chart_names, index=0)

if selected and selected in charts:
    data = charts[selected]
    r    = result_map.get(selected)

    # 지표 요약
    if r:
        c1, c2, c3, c4 = st.columns(4)
        stage_color = STAGE_COLORS.get(r['stage'], "#888")
        c1.markdown(
            f"<div style='background:{stage_color}22;border-left:4px solid {stage_color};"
            f"padding:8px 12px;border-radius:4px'>"
            f"<div style='color:{stage_color};font-weight:bold;font-size:0.8em'>현재 단계</div>"
            f"<div style='font-size:1.3em;font-weight:bold;color:{stage_color}'>{r['stage'] or '-'}</div></div>",
            unsafe_allow_html=True,
        )
        c2.metric("CN (%)",   f"{r['CN']:.1f}")
        c3.metric("경과일",   f"{r['elapsed_days']}일" if r['elapsed_days'] else "-")
        c4.metric(
            "진입일",
            r['stage_change_date'].strftime("%Y-%m-%d")
            if r['stage_change_date'] is not None and pd.notna(r['stage_change_date'])
            else "-",
        )
        st.markdown("")

    # Plotly 차트
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
        subplot_titles=("가격 / MA150", "CN 지표"),
    )

    fig.add_trace(
        go.Scatter(
            x=data['Date'], y=data['Close'],
            name="가격", line=dict(color="#2196F3", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>가격: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data['Date'], y=data['MA150'],
            name="MA150", line=dict(color="#FF9800", width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>MA150: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data['Date'], y=data['CN'],
            name="CN", line=dict(color="#F44336", width=1.5),
            fill="tozeroy", fillcolor="rgba(244,67,54,0.08)",
            hovertemplate="%{x|%Y-%m-%d}<br>CN: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    # CN 기준선
    fig.add_hline(y=30, line_dash="dot", line_color="green",  line_width=1, row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",    line_width=1, row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{selected}", font=dict(size=18)),
        height=580,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(title_text="가격",  row=1, col=1)
    fig.update_yaxes(title_text="CN (%)", row=2, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

# ─── 푸터 ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "본 대시보드는 투자 참고용이며 투자 권유가 아닙니다. "
    "데이터는 yfinance / pykrx 기준이며 실제 시세와 다를 수 있습니다."
)
