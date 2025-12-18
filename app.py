# ============================================================
#  STREAMLIT DASHBOARD - ANALISIS KINERJA KONSELOR MEDIS
#  Header: judul saja (tanpa rentang data & tanpa menu atas)
#  Tema: perpaduan oranye (SIAP Bahagia style)
#  Pie label jelas + Sidebar compact + Cloud ready
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
# PAGE CONFIG (harus paling atas)
# ---------------------------
st.set_page_config(
    page_title="Evaluasi Kinerja Konselor Medis • SIAP BAHAGIA",
    page_icon="🟠",
    layout="wide",
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# ---------------------------
# CSS THEME (orange-soft + readable + compact)
# ---------------------------
st.markdown(
    r"""
<style>
:root{
  --bg: #f6f8fc;
  --panel: rgba(255,255,255,0.94);
  --panel2: rgba(255,255,255,0.86);
  --text: #0f172a;
  --muted: #475569;
  --border: rgba(15, 23, 42, 0.10);
  --shadow: 0 10px 26px rgba(2,6,23,0.08);

  /* ORANGE SIAP BAHAGIA */
  --primary: #f27d20;
  --primary2: rgba(242,125,32,0.16);
  --primaryBorder: rgba(242,125,32,0.30);
  --primaryText: #9a3412;
}

/* background app */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 15% 0%,
    rgba(242,125,32,0.12) 0%,
    #f6f8fc 38%,
    #eef2f7 100%);
}

/* teks main selalu gelap */
section.main, section.main *{
  color: var(--text);
}

/* container */
div.block-container{
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
  max-width: 1400px;
}

/* HERO (judul saja) */
.hero{
  background: linear-gradient(90deg,
    rgba(242,125,32,0.20) 0%,
    rgba(242,125,32,0.12) 55%,
    rgba(255,255,255,0.60) 100%);
  border: 1px solid var(--primaryBorder);
  box-shadow: var(--shadow);
  border-radius: 18px;

  padding: 1.55rem 1.1rem 1.25rem 1.1rem; /* tambah ruang atas */
  margin-bottom: 0.9rem;

  overflow: visible;                       /* pastikan tidak crop isi */
}

.hero-title{
  font-size: 1.65rem;
  font-weight: 950;
  letter-spacing: -0.02em;
  text-align: center;
  color: #111827 !important;

  margin: 0;
  padding-top: 0.25rem;                    /* dorong turun sedikit */
  line-height: 1.25;                       /* cegah teks kepotong */
  white-space: normal;                     /* izinkan turun baris */
  word-break: break-word;
}


/* KPI cards */
.metric-card{
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: 0 8px 18px rgba(2,6,23,0.07);
  border-radius: 16px;
  padding: 0.85rem 0.9rem;
}
.metric-label{
  color: var(--muted) !important;
  font-size: 0.82rem;
  font-weight: 900;
  margin-bottom: 0.2rem;
}
.metric-value{
  color: var(--text) !important;
  font-size: 1.45rem;
  font-weight: 950;
  letter-spacing:-0.02em;
}
.metric-foot{
  color: rgba(71,85,105,0.92) !important;
  font-size: 0.78rem;
  margin-top: 0.18rem;
}

/* Section chip */
.section-chip{
  display:inline-block;
  padding: 0.25rem 0.70rem;
  border-radius: 999px;
  background: rgba(242,125,32,0.10);
  border: 1px solid rgba(242,125,32,0.22);
  color: var(--primaryText) !important;
  font-weight: 950;
  margin: 0.15rem 0 0.75rem 0;
}

/* Card block */
.card{
  background: var(--panel2);
  border: 1px solid var(--border);
  box-shadow: 0 8px 18px rgba(2,6,23,0.06);
  border-radius: 16px;
  padding: 0.85rem 0.95rem 0.55rem 0.95rem;
  margin-bottom: 0.9rem;
}
.card h3{
  margin: 0 0 0.55rem 0;
  color: var(--text) !important;
  font-size: 1.05rem;
}
.small-note{
  color: var(--muted) !important;
  font-size: 0.85rem;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
}
[data-testid="stSidebar"] *{
  color: #e5e7eb !important;
}

/* COMPACT MODE */
html, body, [data-testid="stAppViewContainer"]{
  font-size: 14px;
}
[data-testid="stSidebar"][aria-expanded="true"]{
  min-width: 270px !important;
  max-width: 270px !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown{
  font-size: 0.86rem !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea{
  font-size: 0.90rem !important;
}

/* Dataframe */
div[data-testid="stDataFrame"]{
  background: rgba(255,255,255,0.60);
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(15,23,42,0.10);
}

/* Paksa semua teks di area main jadi gelap */
section.main, section.main *{
  color: #0f172a !important;
}

/* Heading juga dipaksa gelap */
section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6{
  color: #0f172a !important;
}

/* Caption sering jadi pudar, kita tegasin sedikit */
[data-testid="stCaptionContainer"], .stCaption{
  color: #475569 !important;
  opacity: 1 !important;
}

/* Paksa teks markdown di MAIN jadi gelap dan tidak pudar */
section.main div[data-testid="stMarkdownContainer"],
section.main div[data-testid="stMarkdownContainer"] *{
  color: #0f172a !important;
  opacity: 1 !important;
}

/* Bullet list sering paling kena pudar */
section.main div[data-testid="stMarkdownContainer"] ul li,
section.main div[data-testid="stMarkdownContainer"] ol li{
  color: #0f172a !important;
  opacity: 1 !important;
}

/* Kadang Streamlit bungkus markdown pakai class ini */
section.main .stMarkdown,
section.main .stMarkdown *{
  color: #0f172a !important;
  opacity: 1 !important;
}

/* Saat user blok/seleksi teks */
::selection{
  background: rgba(242,125,32,0.35);  /* oranye muda */
  color: #0f172a;                     /* teks tetap gelap */
}
::-moz-selection{
  background: rgba(242,125,32,0.35);
  color: #0f172a;
}

/* Hide footer */
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


def style_pie_autotexts(autotexts):
    """Angka pie chart terlihat jelas (kontras aman)."""
    for t in autotexts:
        t.set_color("#0f172a")
        t.set_fontweight("bold")
        t.set_fontsize(10)
        t.set_bbox(dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            boxstyle="round,pad=0.25"
        ))
def autopct_min_pct(min_pct: float = 3.0):
    """Hanya tampilkan persen jika >= min_pct agar tidak numpuk."""
    def _fmt(pct):
        return f"{pct:.1f}%" if pct >= min_pct else ""
    return _fmt


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))


# ---------------------------
# SIDEBAR: Upload + Menu + Filter
# ---------------------------
safe_sidebar_image("logo_siapbahagia.jpg")

st.sidebar.markdown("## ⚙️ Kontrol")
uploaded = st.sidebar.file_uploader("📥 Upload File Excel", type=["xlsx"])

if uploaded is None:
    # Header judul saja
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">EVALUASI KINERJA KONSELOR MEDIS • SIAP BAHAGIA</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# Load data
df = load_excel(uploaded.getvalue())

# Validasi kolom wajib
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

# Clean + index tanggal
df["Tanggal Pertanyaan"] = pd.to_datetime(df["Tanggal Pertanyaan"], errors="coerce")
df = df.dropna(subset=["Tanggal Pertanyaan"]).copy()
df = df.set_index("Tanggal Pertanyaan")

# Normalisasi teks
for col in ["Konselor", "Jadwal Seharusnya", "Hari Pertanyaan", "Hari Jawab"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Waktu respon numeric
df["Waktu Respon"] = pd.to_numeric(
    df["Waktu Respon"].replace("-", np.nan),
    errors="coerce",
)

# Bulan & Tahun
month_map = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu",
    9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}
df["Bulan Respon"] = df.index.month.map(month_map)
df["Tahun"] = df.index.year

bulan_order = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
bulan_available = [b for b in bulan_order if b in df["Bulan Respon"].unique()]

# Jam
if "Jam Pertanyaan" in df.columns:
    df["Jam"] = pd.to_datetime(df["Jam Pertanyaan"].astype(str), errors="coerce").dt.hour
else:
    df["Jam"] = df.index.hour

# Menu pindah halaman di SIDEBAR (menggantikan tulisan di atas)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Menu")
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Dashboard Global",
        "Dashboard Personal",
        "Time Series Mingguan",
        "Change Point Detection",
        "Analisis Statistik",
    ],
    label_visibility="collapsed",
)

# Filter Global
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Filter Global")
bulan_global = st.sidebar.multiselect("Bulan", options=bulan_available, default=bulan_available)
tahun_global = st.sidebar.multiselect(
    "Tahun",
    options=sorted(df["Tahun"].unique()),
    default=sorted(df["Tahun"].unique()),
)

konselor_list = ["Semua Konselor"] + sorted(df["Konselor"].unique())
konselor_global = st.sidebar.selectbox("Konselor Penjawab", konselor_list)

df_global = df[df["Bulan Respon"].isin(bulan_global) & df["Tahun"].isin(tahun_global)]
if konselor_global != "Semua Konselor":
    df_global = df_global[df_global["Konselor"] == konselor_global]

# Filter Personal
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Filter Personal")
jadwal_list = sorted(df["Jadwal Seharusnya"].unique())
konselor_jadwal = st.sidebar.selectbox("Jadwal Seharusnya", jadwal_list)
bulan_personal = st.sidebar.multiselect("Bulan (Personal)", options=bulan_available, default=bulan_available)

# Export
st.sidebar.markdown("---")
with st.sidebar.expander("⬇️ Export", expanded=False):
    st.download_button(
        "Download data global terfilter (CSV)",
        data=df_global.to_csv(index=True).encode("utf-8"),
        file_name="data_global_terfilter.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------
# HEADER MAIN (judul saja)
# ---------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">EVALUASI KINERJA KONSELOR MEDIS • SIAP BAHAGIA</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# KPI GLOBAL (tetap tampil)
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


# ============================================================
# HALAMAN: DASHBOARD GLOBAL
# ============================================================
if menu == "Dashboard Global":
    st.markdown("<div class='section-chip'>Dashboard Global</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    # Bar Hari
    with colA:
        st.markdown("<div class='card'><h3>Pertanyaan dan Jawaban per Hari</h3>", unsafe_allow_html=True)

        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pert = df_global["Hari Pertanyaan"].value_counts().reindex(order).fillna(0)
        jawab = (
            df_global[df_global["Flag Tidak Terjawab"] == 0]["Hari Pertanyaan"]
            .value_counts().reindex(order).fillna(0)
        )

        fig, ax = plt.subplots(figsize=(7, 4))
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

    # Bar Jam
    with colB:
        st.markdown("<div class='card'><h3>Distribusi Jam Pertanyaan</h3>", unsafe_allow_html=True)
        jam_count = df_global.groupby("Jam").size().sort_index()

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if len(jam_count) > 0:
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

    # Pie Waktu Respon
    with colC:
        st.markdown("<div class='card'><h3>Rata-rata Waktu Respon per Konselor</h3>", unsafe_allow_html=True)

        df_resp = df_global.dropna(subset=["Waktu Respon"])
        df_resp = df_resp[df_resp["Konselor"] != "Tidak Terjawab"]
    
        if df_resp.empty:
            st.warning("Tidak ada data waktu respon pada filter ini.")
        else:
            mean_resp = df_resp.groupby("Konselor")["Waktu Respon"].mean().sort_values(ascending=False)
    
            if mean_resp.sum() <= 0:
                st.warning("Nilai pie chart tidak valid (semua nol).")
            else:
                fig3, ax3 = plt.subplots(figsize=(7, 6.0))
                wedges, texts, autotexts = ax3.pie(
                    mean_resp.values,
                    labels=None,                  # label jangan di slice (biar bersih)
                    autopct=autopct_min_pct(3.0), # persen kecil (<3%) disembunyikan
                    startangle=90,
                    pctdistance=0.68,             # angka agak masuk ke tengah
                    wedgeprops={"linewidth": 1, "edgecolor": "white"},
                )
    
                style_pie_autotexts(autotexts)
                ax3.axis("equal")
    
                ax3.legend(
                    wedges, mean_resp.index,
                    loc="upper center", bbox_to_anchor=(0.5, -0.05),
                    ncol=2, frameon=False
                )
    
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)
    
        st.markdown("</div>", unsafe_allow_html=True)


    # Pie Status
    with colD:
        st.markdown("<div class='card'><h3>Proporsi Sesuai / Tidak Sesuai / Tidak Terjawab</h3>", unsafe_allow_html=True)

        values = [
            int((df_global["Flag Sesuai"] == 1).sum()),
            int((df_global["Flag Tidak Sesuai"] == 1).sum()),
            int((df_global["Flag Tidak Terjawab"] == 1).sum()),
        ]
        labels = ["Sesuai", "Tidak Sesuai", "Tidak Terjawab"]

        if sum(values) == 0:
            st.warning("Tidak ada data status pada filter ini.")
        else:
            fig4, ax4 = plt.subplots(figsize=(7, 6.0))
            wedges, texts, autotexts = ax4.pie(
                values,
                autopct="%1.0f%%",
                startangle=90,
                pctdistance=0.72,
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
            )
            style_pie_autotexts(autotexts)
            ax4.axis("equal")
            ax4.legend(
                wedges, labels,
                loc="upper center", bbox_to_anchor=(0.5, -0.05),
                ncol=3, frameon=False
            )
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        st.markdown("</div>", unsafe_allow_html=True)

    # Workload
    st.markdown("<div class='section-chip'>👥 Analisis Beban Kerja Konselor</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    df_work = df_global[df_global["Konselor"] != "Tidak Terjawab"]
    workload = df_work.groupby("Konselor").size().sort_values(ascending=False)

    if workload.empty:
        st.warning("Tidak ada data workload pada filter ini.")
    else:
        fig_wl, ax_wl = plt.subplots(figsize=(9, 4.2))
        workload_sorted = workload.sort_values()
        bars = ax_wl.barh(workload_sorted.index, workload_sorted.values)

        offset = max(1, int(workload_sorted.values.max() * 0.02))
        for bar in bars:
            ax_wl.text(
                bar.get_width() + offset,
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

    # Produktivitas
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
        st.caption("Respon tinggi + waktu respon rendah → performa lebih optimal.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: DASHBOARD PERSONAL
# ============================================================
elif menu == "Dashboard Personal":
    st.markdown("<div class='section-chip'>Dashboard Personal</div>", unsafe_allow_html=True)

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
        kpi_card("Jumlah Pertanyaan", f"{total_q:,}", "Pada filter personal")
    with k2:
        kpi_card("Jumlah Respon", f"{jumlah_respon:,}", f"Response rate: {rr:.1f}%")
    with k3:
        kpi_card("Tidak Terjawab", f"{tidak_terjawab:,}", "Butuh tindak lanjut")
    with k4:
        kpi_card("Sesuai Jadwal", f"{sesuai:,}", "Jawaban sesuai jadwal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h3>Proporsi Kesesuaian Menjawab (Berdasarkan Jadwal)</h3>", unsafe_allow_html=True)

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

        if penjawab_count.sum() == 0:
            st.warning("Tidak ada data pie pada filter ini.")
        else:
            fig1, ax1 = plt.subplots(figsize=(7, 6.0))
            wedges, texts, autotexts = ax1.pie(
                penjawab_count.values,
                labels=penjawab_count.index,
                autopct=lambda p: f"{int(round(p/100 * penjawab_count.sum()))}",
                startangle=90,
                pctdistance=0.72,
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
            )
            style_pie_autotexts(autotexts)
            ax1.axis("equal")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>Proporsi Sesuai / Tidak Sesuai / Tidak Terjawab</h3>", unsafe_allow_html=True)

        labels2 = ["Sesuai", "Tidak Sesuai", "Tidak Terjawab"]
        values2 = [sesuai, tidak_sesuai, tidak_terjawab]

        if sum(values2) == 0:
            st.warning("Tidak ada data status pada filter ini.")
        else:
            fig2, ax2 = plt.subplots(figsize=(7, 6.0))
            wedges, texts, autotexts = ax2.pie(
                values2,
                labels=labels2,
                autopct=lambda p: f"{int(round(p/100 * sum(values2)))}",
                startangle=90,
                pctdistance=0.72,
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
            )
            style_pie_autotexts(autotexts)
            ax2.axis("equal")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: TIME SERIES
# ============================================================
elif menu == "Time Series Mingguan":
    st.markdown("<div class='section-chip'>Time Series Mingguan</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if df_global.empty:
        st.warning("Tidak ada data global untuk time series.")
    else:
        weekly = df_global.resample("W").size()
        fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
        ax_ts.plot(weekly.index, weekly.values, marker="o", linewidth=2)
        ax_ts.set_title("Total Pertanyaan per Minggu")
        ax_ts.set_ylabel("Jumlah Pertanyaan")
        prettify_ax(ax_ts)
        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: CHANGE POINT
# ============================================================
elif menu == "Change Point Detection":
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
            st.caption(f"Penalty otomatis: {penalty:.2f}")

            algo = rpt.Pelt(model="l2", min_size=4).fit(signal)
            cp = algo.predict(pen=penalty)
            st.write("Index Change Point:", cp)

            fig_cp, ax_cp = plt.subplots(figsize=(12, 4))
            ax_cp.plot(signal, marker="o", label="Total Pertanyaan Mingguan", linewidth=2)

            first = True
            for c in cp[:-1]:
                ax_cp.axvline(c, color="red", linestyle="--",
                             label="Change Point" if first else None, alpha=0.70)
                first = False

            ax_cp.set_title("Change Point Detection - Total Pertanyaan Mingguan")
            ax_cp.set_xlabel("Minggu ke- (index)")
            ax_cp.set_ylabel("Jumlah Pertanyaan")
            ax_cp.legend()
            prettify_ax(ax_cp)
            st.pyplot(fig_cp, use_container_width=True)
            plt.close(fig_cp)

            # segmen numerik
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
                st.info(f"Perubahan rata-rata beban layanan: {pct:.1f}% dari segmen 1 ke segmen 2.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: STATISTIK
# ============================================================
elif menu == "Analisis Statistik":
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
