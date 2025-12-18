# ============================================================
#  STREAMLIT DASHBOARD - ANALISIS KINERJA KONSELOR MEDIS
#  Versi UI lebih rapi + Streamlit Cloud friendly (tanpa login)
# ============================================================

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.stats import kruskal
import scikit_posthocs as sp


# ---------------------------
# PAGE CONFIG (harus di atas)
# ---------------------------
st.set_page_config(
    page_title="Evaluasi Kinerja Konselor Medis",
    page_icon="📊",
    layout="wide",
)


# ---------------------------
# UI THEME (lebih modern)
# ---------------------------
st.markdown(
    """
<style>
:root{
  --bgA: #f8fafc;
  --bgB: #e2e8f0;
  --card: rgba(255,255,255,0.82);
  --text: #0f172a;
  --muted: #475569;
  --primary: #2563eb;
  --accent: #f97316;
  --border: rgba(15, 23, 42, 0.12);
  --shadow: 0 12px 34px rgba(2,6,23,0.10);
}

/* background app */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 12% 10%, var(--bgA) 0%, var(--bgB) 45%, #dbeafe 100%);
}

/* sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
[data-testid="stSidebar"] a { color: #93c5fd !important; }

/* main container width & padding */
div.block-container{
  padding-top: 1.1rem;
  padding-bottom: 2.2rem;
  max-width: 1400px;
}

/* HERO HEADER */
.hero{
  background: linear-gradient(90deg, rgba(37,99,235,0.16) 0%, rgba(249,115,22,0.14) 100%);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: 22px;
  padding: 1.15rem 1.35rem;
  margin-bottom: 1rem;
}
.hero-title{
  font-size: 2.1rem;
  font-weight: 900;
  color: var(--text);
  letter-spacing: -0.02em;
  line-height: 1.15;
  margin: 0;
}
.hero-sub{
  margin-top: 0.35rem;
  color: var(--muted);
  font-size: 1.02rem;
}
.badges{
  display:flex;
  flex-wrap:wrap;
  gap:0.55rem;
  margin-top: 0.75rem;
}
.badge{
  background: rgba(37,99,235,0.12);
  border: 1px solid rgba(37,99,235,0.25);
  color: #1e40af;
  padding: 0.34rem 0.75rem;
  border-radius: 999px;
  font-weight: 800;
  font-size: 0.85rem;
}
.badge.orange{
  background: rgba(249,115,22,0.12);
  border: 1px solid rgba(249,115,22,0.25);
  color: #9a3412;
}

/* KPI cards */
.metric-card{
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: 0 10px 26px rgba(2,6,23,0.07);
  border-radius: 16px;
  padding: 0.95rem 1rem;
}
.metric-label{
  color: var(--muted);
  font-size: 0.9rem;
  font-weight: 800;
  margin-bottom: 0.25rem;
}
.metric-value{
  color: var(--text);
  font-size: 1.75rem;
  font-weight: 950;
  letter-spacing:-0.02em;
}
.metric-foot{
  color: rgba(71,85,105,0.85);
  font-size: 0.85rem;
  margin-top: 0.25rem;
}

/* Section chip */
.section-chip{
  display:inline-block;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: rgba(15,23,42,0.06);
  border: 1px solid rgba(15,23,42,0.10);
  font-weight: 900;
  color: var(--text);
  margin: 0.2rem 0 0.85rem 0;
}

/* Card block */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: 0 10px 26px rgba(2,6,23,0.06);
  border-radius: 18px;
  padding: 0.9rem 1.0rem 0.65rem 1.0rem;
  margin-bottom: 1rem;
}
.card h3{
  margin: 0 0 0.65rem 0;
  color: var(--text);
}
.small-note{
  color: var(--muted);
  font-size: 0.9rem;
}

/* Tabs */
div[data-testid="stTabs"] button{
  border-radius: 999px !important;
  padding: 0.35rem 0.95rem !important;
  font-weight: 900 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
  background: rgba(37,99,235,0.14) !important;
  border: 1px solid rgba(37,99,235,0.25) !important;
}

/* Round inputs a bit */
div[data-testid="stFileUploader"] section { border-radius: 16px !important; }
div[data-testid="stTextInput"] input { border-radius: 14px !important; }

/* Dataframe container */
div[data-testid="stDataFrame"]{
  background: rgba(255,255,255,0.60);
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(15,23,42,0.10);
}

/* Hide Streamlit footer */
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# HELPERS
# ---------------------------
def safe_sidebar_image(path: str, caption: str | None = None):
    p = Path(path)
    if p.exists():
        st.sidebar.image(str(p), use_container_width=True, caption=caption)


def kpi_card(title: str, value: int | float | str, foot: str = ""):
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{title}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-foot">{foot}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height and height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def prettify_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))


# ---------------------------
# SIDEBAR
# ---------------------------
safe_sidebar_image("logo_siapbahagia.jpg")

st.sidebar.markdown("## ⚙️ Kontrol Dashboard")
st.sidebar.caption("Upload data → atur filter → lihat tab analisis.")

uploaded = st.sidebar.file_uploader("📥 Upload File Excel", type=["xlsx"])

with st.sidebar.expander("ℹ️ Catatan format data", expanded=False):
    st.write(
        """
Kolom penting yang dipakai dashboard:
- **Tanggal Pertanyaan** (tanggal/waktu)
- **Konselor**
- **Jadwal Seharusnya**
- **Hari Pertanyaan**
- **Waktu Respon**
- **Flag Sesuai**, **Flag Tidak Sesuai**, **Flag Tidak Terjawab**
(Opsional) **Jam Pertanyaan**
        """.strip()
    )

if uploaded is None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">Evaluasi Kinerja Konselor Medis</div>
  <div class="hero-sub">Silakan upload file Excel dari sidebar untuk mulai analisis.</div>
  <div class="badges">
    <div class="badge">People Analytics</div>
    <div class="badge orange">Time Series</div>
    <div class="badge">Change Point</div>
    <div class="badge orange">Statistik Nonparametrik</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()


# ---------------------------
# LOAD & CLEAN DATA
# ---------------------------
df = load_excel(uploaded.getvalue())

required_cols = [
    "Tanggal Pertanyaan",
    "Konselor",
    "Jadwal Seharusnya",
    "Hari Pertanyaan",
    "Waktu Respon",
    "Flag Sesuai",
    "Flag Tidak Sesuai",
    "Flag Tidak Terjawab",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()

df["Tanggal Pertanyaan"] = pd.to_datetime(df["Tanggal Pertanyaan"], errors="coerce")
df = df.dropna(subset=["Tanggal Pertanyaan"]).copy()
df = df.set_index("Tanggal Pertanyaan")

for col in ["Konselor", "Jadwal Seharusnya", "Hari Pertanyaan", "Hari Jawab"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["Waktu Respon"] = pd.to_numeric(
    df["Waktu Respon"].replace("-", np.nan),
    errors="coerce",
)

# ---------------------------
# BULAN & TAHUN
# ---------------------------
month_map = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu",
    9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}
df["Bulan Respon"] = df.index.month.map(month_map)
df["Tahun"] = df.index.year

bulan_order = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
bulan_available = [b for b in bulan_order if b in df["Bulan Respon"].unique()]

if "Jam Pertanyaan" in df.columns:
    df["Jam"] = pd.to_datetime(df["Jam Pertanyaan"].astype(str), errors="coerce").dt.hour
else:
    df["Jam"] = df.index.hour


# ---------------------------
# HEADER (lebih rapi)
# ---------------------------
min_date = df.index.min().date() if len(df) else "-"
max_date = df.index.max().date() if len(df) else "-"

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">EVALUASI KINERJA KONSELOR MEDIS • SIAP BAHAGIA</div>
  <div class="hero-sub">Rentang data: <b>{min_date}</b> s/d <b>{max_date}</b> • Total baris: <b>{len(df):,}</b></div>
  <div class="badges">
    <div class="badge">Dashboard Global</div>
    <div class="badge orange">Dashboard Personal</div>
    <div class="badge">Time Series Mingguan</div>
    <div class="badge orange">Change Point</div>
    <div class="badge">Uji Kruskal–Wallis + Dunn</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# FILTER GLOBAL (Sidebar)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Filter Global")

bulan_global = st.sidebar.multiselect(
    "Pilih Bulan",
    options=bulan_available,
    default=bulan_available,
)

tahun_global = st.sidebar.multiselect(
    "Pilih Tahun",
    options=sorted(df["Tahun"].unique()),
    default=sorted(df["Tahun"].unique()),
)

konselor_list = ["Semua Konselor"] + sorted(df["Konselor"].unique())
konselor_global = st.sidebar.selectbox("Filter Konselor Penjawab", konselor_list)

df_global = df[df["Bulan Respon"].isin(bulan_global) & df["Tahun"].isin(tahun_global)]
if konselor_global != "Semua Konselor":
    df_global = df_global[df_global["Konselor"] == konselor_global]


# ---------------------------
# FILTER PERSONAL (Sidebar)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Filter Personal")

jadwal_list = sorted(df["Jadwal Seharusnya"].unique())
konselor_jadwal = st.sidebar.selectbox("Pilih Konselor (Jadwal Seharusnya)", jadwal_list)

bulan_personal = st.sidebar.multiselect(
    "Pilih Bulan (Personal)",
    options=bulan_available,
    default=bulan_available,
)

# Export kecil (opsional)
st.sidebar.markdown("---")
with st.sidebar.expander("⬇️ Export (opsional)", expanded=False):
    st.download_button(
        "Download data global terfilter (CSV)",
        data=df_global.to_csv(index=True).encode("utf-8"),
        file_name="data_global_terfilter.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------
# KPI GLOBAL (lebih cantik)
# ---------------------------
total_q_global = int(len(df_global))
total_res_global = int((df_global["Flag Tidak Terjawab"] == 0).sum())
total_not_global = int((df_global["Flag Tidak Terjawab"] == 1).sum())
total_sesuai_global = int((df_global["Flag Sesuai"] == 1).sum())

resp_rate = (total_res_global / total_q_global * 100) if total_q_global else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Jumlah Pertanyaan", f"{total_q_global:,}", "Total pertanyaan pada filter global")
with c2:
    kpi_card("Jumlah Respon", f"{total_res_global:,}", f"Response rate: {resp_rate:.1f}%")
with c3:
    kpi_card("Tidak Terjawab", f"{total_not_global:,}", "Perlu perhatian (SLA / kapasitas)")
with c4:
    kpi_card("Total Bantuan (Sesuai)", f"{total_sesuai_global:,}", "Jawaban sesuai jadwal/ketentuan")

st.markdown("<div class='small-note'>Tip: gunakan filter di sidebar untuk fokus periode/konselor tertentu.</div>", unsafe_allow_html=True)
st.markdown("")


# ---------------------------
# TABS
# ---------------------------
tab_global, tab_personal, tab_ts, tab_cp, tab_stat = st.tabs([
    "📊 Dashboard Global",
    "👤 Dashboard Personal",
    "📈 Time Series Mingguan",
    "🔍 Change Point Detection",
    "📊 Analisis Statistik",
])


# ============================================================
# TAB 1 — DASHBOARD GLOBAL
# ============================================================
with tab_global:
    st.markdown("<div class='section-chip'>Dashboard Global</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    # -------- BAR HARI --------
    with colA:
        st.markdown("<div class='card'><h3>Pertanyaan dan Jawaban per Hari</h3>", unsafe_allow_html=True)

        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pert = df_global["Hari Pertanyaan"].value_counts().reindex(order).fillna(0)
        jawab = (
            df_global[df_global["Flag Tidak Terjawab"] == 0]["Hari Pertanyaan"]
            .value_counts().reindex(order).fillna(0)
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        x = np.arange(len(order))
        width = 0.38

        bars1 = ax.bar(x - width/2, pert.values, width, label="Pertanyaan")
        bars2 = ax.bar(x + width/2, jawab.values, width, label="Jawaban")

        add_bar_labels(ax, bars1)
        add_bar_labels(ax, bars2)

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=18, ha="right")
        ax.legend()
        prettify_ax(ax)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- BAR JAM --------
    with colB:
        st.markdown("<div class='card'><h3>Distribusi Jam Pertanyaan</h3>", unsafe_allow_html=True)

        jam_count = df_global.groupby("Jam").size().sort_index()

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")

        bars = ax2.bar(jam_count.index.astype(int).astype(str), jam_count.values)
        add_bar_labels(ax2, bars)
        ax2.set_xlabel("Jam")
        ax2.set_ylabel("Jumlah Pertanyaan")
        prettify_ax(ax2)

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.markdown("</div>", unsafe_allow_html=True)

    colC, colD = st.columns(2)

    # -------- PIE WAKTU RESPON --------
    with colC:
        st.markdown("<div class='card'><h3>Rata-rata Waktu Respon per Konselor</h3>", unsafe_allow_html=True)

        df_resp = df_global.dropna(subset=["Waktu Respon"])
        df_resp = df_resp[df_resp["Konselor"] != "Tidak Terjawab"]

        if df_resp.empty:
            st.warning("Tidak ada data waktu respon pada filter ini.")
        else:
            mean_resp = df_resp.groupby("Konselor")["Waktu Respon"].mean().sort_values(ascending=False)

            fig3, ax3 = plt.subplots(figsize=(7, 6.8))
            fig3.patch.set_alpha(0)
            ax3.set_facecolor("none")

            wedges, _, _ = ax3.pie(
                mean_resp.values,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
            )
            ax3.axis("equal")

            ax3.legend(
                wedges, mean_resp.index,
                loc="upper center", bbox_to_anchor=(0.5, -0.05),
                ncol=2, frameon=False
            )

            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- PIE STATUS --------
    with colD:
        st.markdown("<div class='card'><h3>Proporsi Sesuai / Tidak Sesuai / Tidak Terjawab</h3>", unsafe_allow_html=True)

        values = [
            int((df_global["Flag Sesuai"] == 1).sum()),
            int((df_global["Flag Tidak Sesuai"] == 1).sum()),
            int((df_global["Flag Tidak Terjawab"] == 1).sum()),
        ]
        labels = ["Sesuai", "Tidak Sesuai", "Tidak Terjawab"]

        fig4, ax4 = plt.subplots(figsize=(7, 6.8))
        fig4.patch.set_alpha(0)
        ax4.set_facecolor("none")

        wedges, _, _ = ax4.pie(
            values,
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        ax4.axis("equal")
        ax4.legend(
            wedges, labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.05),
            ncol=3, frameon=False
        )

        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # ANALISIS BEBAN KERJA
    # ============================================================
    st.markdown("<div class='section-chip'>👥 Analisis Beban Kerja Konselor</div>", unsafe_allow_html=True)

    df_work = df_global[df_global["Konselor"] != "Tidak Terjawab"]
    workload = df_work.groupby("Konselor").size().sort_values(ascending=False)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if workload.empty:
        st.warning("Tidak ada data workload pada filter ini.")
    else:
        fig_wl, ax_wl = plt.subplots(figsize=(9, 4.3))
        fig_wl.patch.set_alpha(0)
        ax_wl.set_facecolor("none")

        workload_sorted = workload.sort_values()
        bars = ax_wl.barh(workload_sorted.index, workload_sorted.values)

        for bar in bars:
            ax_wl.text(
                bar.get_width() + max(1, workload_sorted.values.max() * 0.01),
                bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width())}",
                va="center",
                fontsize=9,
            )

        ax_wl.set_xlabel("Jumlah Pertanyaan")
        ax_wl.set_title("Distribusi Beban Pertanyaan per Konselor")
        prettify_ax(ax_wl)

        plt.tight_layout()
        st.pyplot(fig_wl, use_container_width=True)
        plt.close(fig_wl)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # ANALISIS PRODUKTIVITAS
    # ============================================================
    st.markdown("<div class='section-chip'>⚙️ Analisis Produktivitas Konselor</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    df_prod = df_global[
        (df_global["Konselor"] != "Tidak Terjawab") &
        (df_global["Flag Tidak Terjawab"] == 0)
    ]

    if df_prod.empty:
        st.warning("Tidak ada data produktivitas (respon) pada filter ini.")
    else:
        produktivitas = (
            df_prod.groupby("Konselor")
            .agg(
                Jumlah_Respon=("Flag Tidak Terjawab", "count"),
                Rata_Waktu_Respon=("Waktu Respon", "mean")
            )
            .round(2)
            .sort_values("Jumlah_Respon", ascending=False)
        )

        st.dataframe(produktivitas, use_container_width=True)

        st.caption(
            "Produktivitas diukur dari jumlah respon dan kecepatan respon. "
            "Respon tinggi + waktu respon rendah → performa lebih optimal."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # TIME SERIES
    # ============================================================
    st.markdown("<div class='section-chip'>📈 Tren Pertanyaan Mingguan</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    weekly = df_global.resample("W").size()

    if weekly.empty:
        st.warning("Tidak ada data mingguan pada filter ini.")
    else:
        fig_ts, ax_ts = plt.subplots(figsize=(14, 4))
        fig_ts.patch.set_alpha(0)
        ax_ts.set_facecolor("none")

        ax_ts.plot(weekly.index, weekly.values, marker="o", linewidth=2)
        ax_ts.set_ylabel("Jumlah Pertanyaan")
        prettify_ax(ax_ts)

        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # CHANGE POINT (ringkas)
    # ============================================================
    st.markdown("<div class='section-chip'>🔍 Change Point Detection (Mingguan)</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    weekly_counts = df_global.resample("W").size()
    signal = weekly_counts.fillna(0).values
    n = len(signal)

    if n < 6:
        st.warning("Data mingguan tidak cukup untuk deteksi change point.")
    else:
        penalty = np.log(n) * np.var(signal)

        algo = rpt.Pelt(model="l2", min_size=4).fit(signal)
        cp = algo.predict(pen=penalty)

        fig_cp, ax_cp = plt.subplots(figsize=(14, 4))
        fig_cp.patch.set_alpha(0)
        ax_cp.set_facecolor("none")

        ax_cp.plot(signal, marker="o", linewidth=2, label="Total Pertanyaan Mingguan")
        for i, c in enumerate(cp[:-1]):
            ax_cp.axvline(c, color="red", linestyle="--", alpha=0.75,
                         label="Change Point" if i == 0 else None)

        ax_cp.set_xlabel("Minggu ke- (index)")
        ax_cp.set_ylabel("Jumlah Pertanyaan")
        ax_cp.legend()
        prettify_ax(ax_cp)

        st.caption(f"Penalty otomatis: {penalty:.2f} • Change point index: {cp}")
        st.pyplot(fig_cp, use_container_width=True)
        plt.close(fig_cp)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 2 — DASHBOARD PERSONAL
# ============================================================
with tab_personal:
    st.markdown("<div class='section-chip'>Dashboard Personal (Berdasarkan Jadwal)</div>", unsafe_allow_html=True)

    df_personal = df[
        (df["Jadwal Seharusnya"] == konselor_jadwal) &
        (df["Bulan Respon"].isin(bulan_personal)) &
        (df["Tahun"].isin(tahun_global))
    ].copy()

    if df_personal.empty:
        st.warning("Tidak ada data untuk konselor tersebut (cek filter bulan/tahun).")
        st.stop()

    st.markdown(f"**Konselor (Jadwal Seharusnya): {konselor_jadwal}**")

    def status_row(r):
        if int(r.get("Flag Tidak Terjawab", 0)) == 1:
            return "Tidak Terjawab"
        if int(r.get("Flag Sesuai", 0)) == 1:
            return "Sesuai"
        if int(r.get("Flag Tidak Sesuai", 0)) == 1:
            return "Tidak Sesuai"
        return "Tidak Terjawab"

    df_personal["Status"] = df_personal.apply(status_row, axis=1)

    total_q = len(df_personal)
    sesuai = int((df_personal["Status"] == "Sesuai").sum())
    tidak_sesuai = int((df_personal["Status"] == "Tidak Sesuai").sum())
    tidak_terjawab = int((df_personal["Status"] == "Tidak Terjawab").sum())
    jumlah_respon = sesuai + tidak_sesuai

    rr = (jumlah_respon / total_q * 100) if total_q else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Jumlah Pertanyaan (jadwal ini)", f"{total_q:,}", "Pada bulan & tahun terpilih")
    with k2:
        kpi_card("Jumlah Respon", f"{jumlah_respon:,}", f"Response rate: {rr:.1f}%")
    with k3:
        kpi_card("Tidak Terjawab", f"{tidak_terjawab:,}", "Butuh tindak lanjut")
    with k4:
        kpi_card("Jawaban Sesuai Jadwal", f"{sesuai:,}", "Kesesuaian dari jadwal ini")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h3>Proporsi Kesesuaian Menjawab Berdasarkan Jadwal</h3>", unsafe_allow_html=True)

        df_pie = df_personal[
            (df_personal["Flag Sesuai"] == 1) |
            (df_personal["Flag Tidak Sesuai"] == 1) |
            (df_personal["Flag Tidak Terjawab"] == 1)
        ].copy()

        def penjawab_valid(row):
            if row["Flag Sesuai"] == 1:
                return konselor_jadwal
            if row["Flag Tidak Sesuai"] == 1:
                return row["Konselor"]
            return "Tidak Terjawab"

        df_pie["Penjawab"] = df_pie.apply(penjawab_valid, axis=1)
        penjawab_count = df_pie["Penjawab"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(7, 6.2))
        fig1.patch.set_alpha(0)
        ax1.set_facecolor("none")
        ax1.pie(
            penjawab_count.values,
            labels=penjawab_count.index,
            autopct=lambda p: f"{int(round(p/100 * penjawab_count.sum()))}",
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        ax1.axis("equal")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>Proporsi Sesuai / Tidak Sesuai / Tidak Terjawab</h3>", unsafe_allow_html=True)

        labels2 = ["Sesuai", "Tidak Sesuai", "Tidak Terjawab"]
        values2 = [sesuai, tidak_sesuai, tidak_terjawab]

        fig2, ax2 = plt.subplots(figsize=(7, 6.2))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        ax2.pie(
            values2,
            labels=labels2,
            autopct=lambda p: f"{int(round(p/100 * sum(values2)))}",
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        ax2.axis("equal")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 3 — TIME SERIES
# ============================================================
with tab_ts:
    st.markdown("<div class='section-chip'>Time Series Mingguan</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if df_global.empty:
        st.warning("Tidak ada data global untuk time series.")
    else:
        weekly = df_global.resample("W").size()
        fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
        fig_ts.patch.set_alpha(0)
        ax_ts.set_facecolor("none")

        ax_ts.plot(weekly.index, weekly.values, marker="o", linewidth=2)
        ax_ts.set_title("Total Pertanyaan per Minggu")
        ax_ts.set_ylabel("Jumlah Pertanyaan")
        prettify_ax(ax_ts)

        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 4 — CHANGE POINT
# ============================================================
with tab_cp:
    st.markdown("<div class='section-chip'>Change Point Detection</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if df_global.empty:
        st.warning("Tidak ada data global untuk change point.")
    else:
        weekly_counts = df_global.resample("W").size()
        signal = weekly_counts.fillna(0).values
        n = len(signal)

        if n < 6:
            st.warning("Data mingguan tidak cukup untuk deteksi change point.")
        else:
            variance = np.var(signal)
            penalty = 1.0 * np.log(n) * variance
            st.caption(f"Penalty otomatis digunakan: {penalty:.2f}")

            algo = rpt.Pelt(model="l2", min_size=4).fit(signal)
            cp = algo.predict(pen=penalty)
            st.write("Index Change Point:", cp)

            fig_cp, ax_cp = plt.subplots(figsize=(12, 4))
            fig_cp.patch.set_alpha(0)
            ax_cp.set_facecolor("none")

            ax_cp.plot(signal, marker="o", label="Total Pertanyaan Mingguan", linewidth=2)

            first = True
            for c in cp[:-1]:
                ax_cp.axvline(c, color="red", linestyle="--",
                             label="Change Point" if first else None, alpha=0.75)
                first = False

            ax_cp.set_title("Change Point Detection - Total Pertanyaan Mingguan")
            ax_cp.set_xlabel("Minggu ke- (index)")
            ax_cp.set_ylabel("Jumlah Pertanyaan")
            ax_cp.legend()
            prettify_ax(ax_cp)

            st.pyplot(fig_cp, use_container_width=True)
            plt.close(fig_cp)

            segments = []
            start = 0
            for i, end in enumerate(cp):
                seg = signal[start:end]
                segments.append({
                    "Segmen": i + 1,
                    "Minggu Mulai": start + 1,
                    "Minggu Akhir": end,
                    "Durasi (minggu)": end - start,
                    "Rata-rata / minggu": round(float(seg.mean()), 2) if len(seg) else 0.0,
                    "Total Pertanyaan": int(seg.sum()) if len(seg) else 0,
                    "Std Deviasi": round(float(seg.std()), 2) if len(seg) else 0.0,
                })
                start = end

            df_segments = pd.DataFrame(segments)

            st.subheader("Analisis Numerik per Segmen")
            st.dataframe(df_segments, use_container_width=True)

            if len(df_segments) >= 2 and df_segments.loc[0, "Rata-rata / minggu"] != 0:
                delta = df_segments.loc[1, "Rata-rata / minggu"] - df_segments.loc[0, "Rata-rata / minggu"]
                pct = (delta / df_segments.loc[0, "Rata-rata / minggu"]) * 100
                st.info(f"Terjadi perubahan rata-rata beban layanan sebesar {pct:.1f}% dari segmen 1 ke segmen 2.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 5 — STATISTIK
# ============================================================
with tab_stat:
    st.markdown("<div class='section-chip'>Analisis Statistik</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    df_k = df_global.dropna(subset=["Waktu Respon"])
    df_k = df_k[df_k["Konselor"] != "Tidak Terjawab"]

    if df_k["Konselor"].nunique() < 2:
        st.warning("Minimal perlu 2 konselor dengan data respon.")
        st.stop()

    groups = df_k.groupby("Konselor")["Waktu Respon"].apply(list)
    stat, p = kruskal(*groups)

    st.subheader("📊 Hasil Uji Kruskal–Wallis")
    m1, m2 = st.columns(2)
    with m1:
        kpi_card("Statistik H", f"{stat:.2f}", "Uji beda median antar konselor")
    with m2:
        kpi_card("p-value", f"{p:.5g}", "p < 0.05 → beda signifikan")

    st.subheader("📋 Uji Lanjutan (Dunn Post-hoc)")
    posthoc = sp.posthoc_dunn(
        df_k,
        val_col="Waktu Respon",
        group_col="Konselor",
        p_adjust="bonferroni",
    )
    st.dataframe(posthoc, use_container_width=True)

    alpha = 0.05
    signif_pairs = []
    for i in posthoc.index:
        for j in posthoc.columns:
            if i != j and posthoc.loc[i, j] < alpha:
                signif_pairs.append(tuple(sorted([i, j])))
    signif_pairs = sorted(set(signif_pairs))

    st.markdown("---")
    st.subheader("📝 Interpretasi")

    if p < alpha:
        st.success(
            f"Uji **Kruskal–Wallis** menunjukkan **perbedaan kinerja yang signifikan** "
            f"antar konselor (H = {stat:.2f}; p-value = {p:.5g})."
        )
    else:
        st.info(
            f"Uji **Kruskal–Wallis** menunjukkan **tidak terdapat perbedaan kinerja yang signifikan** "
            f"antar konselor (H = {stat:.2f}; p-value = {p:.5g})."
        )

    if signif_pairs:
        st.markdown("Pasangan konselor yang berbeda signifikan (p < 0.05):")
        for a, b in signif_pairs:
            st.markdown(f"- **{a}** vs **{b}**")
    else:
        st.markdown("Tidak ada pasangan konselor yang berbeda signifikan (p ≥ 0.05).")

    st.caption("Catatan: interpretasi statistik sebaiknya dilengkapi konteks operasional (beban kerja, jadwal, kompleksitas kasus).")

    st.markdown("</div>", unsafe_allow_html=True)
