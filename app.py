%%writefile dashboard_bkkbn.py
# ============================================================
#      STREAMLIT DASHBOARD - ANALISIS KINERJA KONSELOR MEDIS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.stats import kruskal
import scikit_posthocs as sp

st.cache_data.clear()
st.cache_resource.clear()

# ------------------------------------------------------------
# HELPER: LABEL ANGKA DI BAR
# ------------------------------------------------------------
def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )
            
# ------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------
st.set_page_config(page_title="Evaluasi Kinerja Konselor Medis", layout="wide")

st.sidebar.image(
    "logo_siapbahagia.jpg",
    use_container_width=True
)

# ------------------------------------------------------------
# CSS THEME
# ------------------------------------------------------------
st.markdown("""
<style>
body { background-color: #e0c7b4; }
.main { background-color: #111111; color: white; }
.header-container {
    background-color: #f27d20; padding: 15px 30px; border-radius: 20px;
    text-align: center; color: white; font-size: 24px; font-weight: bold;
}
.kpi-card {
    background-color: white; color: black;
    padding: 15px; border-radius: 12px; text-align: center;
}
.kpi-title { font-size: 14px; color: #444; }
.kpi-value { font-size: 26px; font-weight: bold; }
.subheader-box {
    background-color: white; color: black;
    padding: 8px 15px; border-radius: 10px;
    margin-bottom: 15px; display: inline-block;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown(
    '<div class="header-container">EVALUASI KINERJA KONSELOR MEDIS SIAP BAHAGIA</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# UPLOAD FILE
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader("📥 Upload File", type=["xlsx"])
if uploaded is None:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# ------------------------------------------------------------
# LOAD & CLEAN DATA
# ------------------------------------------------------------
df = pd.read_excel(uploaded)

# Konversi tanggal
df["Tanggal Pertanyaan"] = pd.to_datetime(df["Tanggal Pertanyaan"], errors="coerce")
df = df.dropna(subset=["Tanggal Pertanyaan"])
df = df.set_index("Tanggal Pertanyaan")

# Normalisasi kolom teks
for col in ["Konselor", "Jadwal Seharusnya", "Hari Pertanyaan", "Hari Jawab"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Konversi angka respon
df["Waktu Respon"] = pd.to_numeric(
    df["Waktu Respon"].replace("-", np.nan),
    errors="coerce"
)

# ============================================================
# BULAN & TAHUN (SATU KALI SAJA)
# ============================================================
month_map = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu",
    9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"
}

df["Bulan Respon"] = df.index.month.map(month_map)
df["Tahun"] = df.index.year

bulan_order = [
    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
    "Jul", "Agu", "Sep", "Okt", "Nov", "Des"
]

bulan_available = [
    b for b in bulan_order
    if b in df["Bulan Respon"].unique()
]

# Jam pertanyaan
if "Jam Pertanyaan" in df.columns:
    df["Jam"] = pd.to_datetime(df["Jam Pertanyaan"].astype(str), errors="coerce").dt.hour
else:
    df["Jam"] = df.index.hour

# ------------------------------------------------------------
# FILTER GLOBAL
# ------------------------------------------------------------
st.sidebar.markdown("### 🌍 Filter Global")

bulan_available = [b for b in bulan_order if b in df["Bulan Respon"].unique()]

bulan_global = st.sidebar.multiselect(
    "Pilih Bulan",
    options=bulan_available,
    default=bulan_available
)

tahun_global = st.sidebar.multiselect(
    "Pilih Tahun",
    options=sorted(df["Tahun"].unique()),
    default=sorted(df["Tahun"].unique())
)

konselor_list = ["Semua Konselor"] + sorted(df["Konselor"].unique())
konselor_global = st.sidebar.selectbox("Filter Konselor Penjawab", konselor_list)

df_global = df[df["Bulan Respon"].isin(bulan_global) & df["Tahun"].isin(tahun_global)]
if konselor_global != "Semua Konselor":
    df_global = df_global[df_global["Konselor"] == konselor_global]

# ------------------------------------------------------------
# FILTER PERSONAL (Jadwal)
# ------------------------------------------------------------
st.sidebar.markdown("### 👤 Filter Personal")

jadwal_list = sorted(df["Jadwal Seharusnya"].unique())
konselor_jadwal = st.sidebar.selectbox(
    "Pilih Konselor (Jadwal Seharusnya)",
    jadwal_list
)

bulan_personal = st.sidebar.multiselect(
    "Pilih Bulan (Personal)",
    options=bulan_available,
    default=bulan_available
)


# ------------------------------------------------------------
# KPI GLOBAL
# ------------------------------------------------------------
total_q_global = len(df_global)
total_res_global = (df_global["Flag Tidak Terjawab"] == 0).sum()
total_not_global = (df_global["Flag Tidak Terjawab"] == 1).sum()
total_sesuai_global = (df_global["Flag Sesuai"] == 1).sum()

col_g1, col_g2, col_g3, col_g4 = st.columns(4)
for col, title, value in [
    (col_g1, "Jumlah Pertanyaan", total_q_global),
    (col_g2, "Jumlah Respon", total_res_global),
    (col_g3, "Tidak Terjawab", total_not_global),
    (col_g4, "Total Bantuan (Sesuai)", total_sesuai_global),
]:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_global, tab_personal, tab_ts, tab_cp, tab_stat = st.tabs([
    "📊 Dashboard Global",
    "👤 Dashboard Personal",
    "📈 Time Series Mingguan",
    "🔍 Change Point Detection",
    "📊 Analisis Statistik",
])

# ============================================================
#  TAB 1 — DASHBOARD GLOBAL
# ============================================================
with tab_global:
    st.markdown('<div class="subheader-box">Dashboard Global</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # -------- BAR HARI --------
    with col1:
        st.subheader("Pertanyaan dan Jawaban per Hari")

        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pert = df_global["Hari Pertanyaan"].value_counts().reindex(order).fillna(0)
        jawab = df_global[df_global["Flag Tidak Terjawab"] == 0]["Hari Pertanyaan"].value_counts().reindex(order).fillna(0)

        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(len(order))
        width = 0.35

        bars1 = ax.bar(x - width/2, pert, width, label="Pertanyaan")
        bars2 = ax.bar(x + width/2, jawab, width, label="Jawaban")

        add_bar_labels(ax, bars1)
        add_bar_labels(ax, bars2)

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------- BAR JAM --------
    with col2:
        st.subheader("Distribusi Jam Pertanyaan")

        jam_count = df_global.groupby("Jam").size()

        fig2, ax2 = plt.subplots(figsize=(6,4))
        bars = ax2.bar(jam_count.index.astype(str), jam_count.values)
        add_bar_labels(ax2, bars)

        ax2.set_xlabel("Jam")
        ax2.set_ylabel("Jumlah Pertanyaan")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    col3, col4 = st.columns(2)

    # -------- PIE WAKTU RESPON --------
    with col3:
        st.subheader("Rata-rata Waktu Respon per Konselor")

        df_resp = df_global.dropna(subset=["Waktu Respon"])
        df_resp = df_resp[df_resp["Konselor"] != "Tidak Terjawab"]

        if not df_resp.empty:
            mean_resp = df_resp.groupby("Konselor")["Waktu Respon"].mean()

            fig3, ax3 = plt.subplots(figsize=(7,7))
            wedges, _, _ = ax3.pie(mean_resp.values, autopct="%1.1f%%", startangle=90)
            ax3.axis("equal")

            ax3.legend(
                wedges, mean_resp.index,
                loc="upper center", bbox_to_anchor=(0.5, -0.05),
                ncol=2, frameon=False
            )

            st.pyplot(fig3)
            plt.close(fig3)

    # -------- PIE STATUS --------
    with col4:
        st.subheader("Proporsi Sesuai / Tidak Sesuai / Tidak Terjawab")

        values = [
            (df_global["Flag Sesuai"] == 1).sum(),
            (df_global["Flag Tidak Sesuai"] == 1).sum(),
            (df_global["Flag Tidak Terjawab"] == 1).sum(),
        ]

        labels = ["Sesuai", "Tidak Sesuai", "Tidak Terjawab"]

        fig4, ax4 = plt.subplots(figsize=(7,7))
        wedges, _, _ = ax4.pie(values, autopct="%1.0f%%", startangle=90)
        ax4.axis("equal")

        ax4.legend(
            wedges, labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.05),
            ncol=3, frameon=False
        )

        st.pyplot(fig4)
        plt.close(fig4)

    # ============================================================
    # ANALISIS BEBAN KERJA KONSELOR
    # ============================================================
    st.markdown("---")
    st.markdown("### 👥 Analisis Beban Kerja Konselor")

    workload = (
        df_global[df_global["Konselor"] != "Tidak Terjawab"]
        .groupby("Konselor")
        .size()
        .sort_values(ascending=False)
    )

    fig_wl, ax_wl = plt.subplots(figsize=(8,4))

    workload_sorted = workload.sort_values()

    bars = ax_wl.barh(workload_sorted.index, workload_sorted.values)

    for bar in bars:
        ax_wl.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height()/2,
            f"{int(bar.get_width())}",
            va="center"
        )

    ax_wl.set_xlabel("Jumlah Pertanyaan")
    ax_wl.set_title("Distribusi Beban Pertanyaan per Konselor")

    plt.tight_layout()
    st.pyplot(fig_wl)
    plt.close(fig_wl)

    # ============================================================
    # ANALISIS PRODUKTIVITAS KONSELOR
    # ============================================================
    st.markdown("### ⚙️ Analisis Produktivitas Konselor")

    df_prod = df_global[
        (df_global["Konselor"] != "Tidak Terjawab") &
        (df_global["Flag Tidak Terjawab"] == 0)
    ]

    produktivitas = (
        df_prod.groupby("Konselor")
        .agg(
            Jumlah_Respon=("Flag Tidak Terjawab", "count"),
            Rata_Waktu_Respon=("Waktu Respon", "mean")
        )
        .round(2)
    )

    st.dataframe(produktivitas, use_container_width=True)

    st.caption(
        "Produktivitas diukur dari jumlah respon dan kecepatan respon. "
        "Konselor dengan respon tinggi dan waktu respon rendah menunjukkan performa optimal."
    )

    # ============================================================
    # TIME SERIES
    # ============================================================
    st.markdown("---")
    st.markdown("### 📈 Tren Pertanyaan Mingguan")

    weekly = df_global.resample("W").size()

    fig_ts, ax_ts = plt.subplots(figsize=(14,4))
    ax_ts.plot(weekly.index, weekly.values, marker="o")
    ax_ts.set_ylabel("Jumlah Pertanyaan")
    ax_ts.grid(alpha=0.3)

    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # ============================================================
    # CHANGE POINT DETECTION
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔍 Change Point Detection (Mingguan)")

    weekly_counts = df_global.resample("W").size()
    signal = weekly_counts.fillna(0).values
    n = len(signal)

    if n < 6:
        st.warning("Data mingguan tidak cukup untuk deteksi change point.")
    else:
        # Penalty otomatis
        penalty = np.log(n) * np.var(signal)

        # Deteksi change point
        algo = rpt.Pelt(model="l2", min_size=4).fit(signal)
        cp = algo.predict(pen=penalty)

        # Plot grafik
        fig_cp, ax_cp = plt.subplots(figsize=(14, 4))
        ax_cp.plot(
            signal,
            marker="o",
            linewidth=2,
            label="Total Pertanyaan Mingguan"
        )

        for i, c in enumerate(cp[:-1]):
            ax_cp.axvline(
                c,
                color="red",
                linestyle="--",
                alpha=0.8,
                label="Change Point" if i == 0 else None
            )

        ax_cp.set_xlabel("Minggu ke-")
        ax_cp.set_ylabel("Jumlah Pertanyaan")
        ax_cp.legend()
        ax_cp.grid(alpha=0.3)

        st.pyplot(fig_cp, use_container_width=True)
        plt.close(fig_cp)


# ============================================================
#  TAB 2 — DASHBOARD PERSONAL
# ============================================================
with tab_personal:
    st.markdown(
        '<div class="subheader-box">Dashboard Personal Konselor (Berdasarkan Jadwal)</div>',
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------
    # FILTER PERSONAL YANG BENAR
    # ------------------------------------------------------------
    # Pakai filter tahun_global + bulan_personal supaya tidak "melebar" jadi 100 data
    df_personal = df[
        (df["Jadwal Seharusnya"] == konselor_jadwal) &
        (df["Bulan Respon"].isin(bulan_personal)) &
        (df["Tahun"].isin(tahun_global))
    ].copy()

    if df_personal.empty:
        st.warning("Tidak ada data untuk konselor tersebut (cek filter bulan/tahun).")
        st.stop()

    st.markdown(f"**Konselor (Jadwal Seharusnya): {konselor_jadwal}**")

    # ------------------------------------------------------------
    # NORMALISASI STATUS (BIAR KPI & PIE SELALU KONSISTEN)
    # Prioritas: Tidak Terjawab > Sesuai > Tidak Sesuai
    # ------------------------------------------------------------
    def status_row(r):
        if int(r.get("Flag Tidak Terjawab", 0)) == 1:
            return "Tidak Terjawab"
        if int(r.get("Flag Sesuai", 0)) == 1:
            return "Sesuai"
        if int(r.get("Flag Tidak Sesuai", 0)) == 1:
            return "Tidak Sesuai"
        return "Tidak Terjawab"

    df_personal["Status"] = df_personal.apply(status_row, axis=1)

    # ------------------------------------------------------------
    # KPI PERSONAL (DIAMBIL DARI STATUS -> KONSISTEN)
    # ------------------------------------------------------------
    total_q = len(df_personal)
    sesuai = int((df_personal["Status"] == "Sesuai").sum())
    tidak_sesuai = int((df_personal["Status"] == "Tidak Sesuai").sum())
    tidak_terjawab = int((df_personal["Status"] == "Tidak Terjawab").sum())
    jumlah_respon = sesuai + tidak_sesuai

    c1, c2, c3, c4 = st.columns(4)
    for col, title, val in [
        (c1, "Jumlah Pertanyaan (pada jadwal ini)", total_q),
        (c2, "Jumlah Respon", jumlah_respon),
        (c3, "Tidak Terjawab", tidak_terjawab),
        (c4, "Jawaban Sesuai Jadwal", sesuai),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # =============================================================
    # PIE 1 — PROPORSI KESESUAIAN MENJAWAB BERDASARKAN JADWAL
    # (SUMBER = FLAG, TOTAL HARUS = KPI PERSONAL)
    # =============================================================
    st.subheader("Proporsi Kesesuaian Menjawab Berdasarkan Jadwal")

    # 🔒 Ambil hanya baris yang memang termasuk KPI
    df_pie = df_personal[
        (df_personal["Flag Sesuai"] == 1) |
        (df_personal["Flag Tidak Sesuai"] == 1) |
        (df_personal["Flag Tidak Terjawab"] == 1)
    ].copy()

    # Tentukan penjawab BERDASARKAN FLAG
    def penjawab_valid(row):
        if row["Flag Sesuai"] == 1:
            return konselor_jadwal
        if row["Flag Tidak Sesuai"] == 1:
            return row["Konselor"]
        return "Tidak Terjawab"

    df_pie["Penjawab"] = df_pie.apply(penjawab_valid, axis=1)

    penjawab_count = df_pie["Penjawab"].value_counts()

    # 🔍 VALIDASI (INI WAJIB = total_q)

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.pie(
        penjawab_count.values,
        labels=penjawab_count.index,
        autopct=lambda p: f"{int(round(p/100 * penjawab_count.sum()))}"
    )
    ax1.axis("equal")
    st.pyplot(fig1)
    plt.close(fig1)


    # =============================================================
    # PIE 2 — SESUAI / TIDAK SESUAI / TIDAK TERJAWAB (MURNI FLAG)
    # =============================================================
    st.subheader("Proporsi Sesuai / Tidak Sesuai / Tidak Terjawab")

    labels2 = ["Sesuai", "Tidak Sesuai", "Tidak Terjawab"]
    values2 = [
        sesuai,
        tidak_sesuai,
        tidak_terjawab
    ]

# 🔍 VALIDASI

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.pie(
        values2,
        labels=labels2,
        autopct=lambda p: f"{int(round(p/100 * sum(values2)))}"
    )
    ax2.axis("equal")
    st.pyplot(fig2)
    plt.close(fig2)


# ============================================================
# TAB 3 — TIME SERIES MINGGUAN
# ============================================================
with tab_ts:
    st.markdown(
        '<div class="subheader-box">Time Series Mingguan</div>',
        unsafe_allow_html=True
    )

    if df_global.empty:
        st.warning("Tidak ada data global untuk time series.")
    else:
        weekly = df_global.resample("W").size()
        fig_ts, ax_ts = plt.subplots(figsize=(8, 4))
        ax_ts.plot(weekly.index, weekly.values, marker="o")
        ax_ts.set_title("Total Pertanyaan per Minggu")
        ax_ts.set_ylabel("Jumlah Pertanyaan")
        st.pyplot(fig_ts)
        plt.close(fig_ts)

        
# ============================================================
# TAB 4 — CHANGE POINT
# ============================================================
with tab_cp:
    st.markdown(
        '<div class="subheader-box">Change Point Detection</div>',
        unsafe_allow_html=True
    )

    if df_global.empty:
        st.warning("Tidak ada data global untuk change point.")
    else:
        # ===============================
        # 1. Agregasi mingguan
        # ===============================
        weekly_counts = df_global.resample("W").size()
        signal = weekly_counts.fillna(0).values
        n = len(signal)

        if n < 6:
            st.warning("Data mingguan tidak cukup untuk deteksi change point.")
        else:
            # ===============================
            # 2. Penalty otomatis
            # ===============================
            variance = np.var(signal)
            penalty = 1.0 * np.log(n) * variance

            st.caption(f"Penalty otomatis digunakan: {round(penalty, 2)}")

            # ===============================
            # 3. Change Point Detection
            # ===============================
            algo = rpt.Pelt(
                model="l2",
                min_size=4
            ).fit(signal)

            cp = algo.predict(pen=penalty)
            st.write("Index Change Point:", cp)

            # ===============================
            # 4. Visualisasi
            # ===============================
            fig_cp, ax_cp = plt.subplots(figsize=(10, 4))
            ax_cp.plot(signal, marker="o", label="Total Pertanyaan Mingguan")

            first = True
            for c in cp[:-1]:
                ax_cp.axvline(
                    c,
                    color="red",
                    linestyle="--",
                    label="Change Point" if first else None
                )
                first = False

            ax_cp.set_title("Change Point Detection - Total Pertanyaan Mingguan")
            ax_cp.set_xlabel("Minggu ke-")
            ax_cp.set_ylabel("Jumlah Pertanyaan")
            ax_cp.legend()
            st.pyplot(fig_cp)
            plt.close(fig_cp)

            # ===============================
            # 5. Analisis numerik per segmen
            # ===============================
            segments = []
            start = 0

            for i, end in enumerate(cp):
                seg = signal[start:end]

                segments.append({
                    "Segmen": i + 1,
                    "Minggu Mulai": start + 1,
                    "Minggu Akhir": end,
                    "Durasi (minggu)": end - start,
                    "Rata-rata / minggu": round(seg.mean(), 2),
                    "Total Pertanyaan": int(seg.sum()),
                    "Std Deviasi": round(seg.std(), 2)
                })

                start = end

            df_segments = pd.DataFrame(segments)

            st.subheader("Analisis Numerik per Segmen")
            st.dataframe(df_segments, use_container_width=True)

            # ===============================
            # 6. Insight otomatis (opsional tapi berguna)
            # ===============================
            if len(df_segments) >= 2:
                delta = (
                    df_segments.loc[1, "Rata-rata / minggu"]
                    - df_segments.loc[0, "Rata-rata / minggu"]
                )
                pct = (delta / df_segments.loc[0, "Rata-rata / minggu"]) * 100

                st.info(
                    f"Terjadi perubahan rata-rata beban layanan sebesar "
                    f"{pct:.1f}% dari segmen 1 ke segmen 2."
                )



# ============================================================
# TAB 5 — STATISTIK NONPARAMETRIK 
# ============================================================
with tab_stat:
    st.markdown(
        '<div class="subheader-box">Analisis Statistik</div>',
        unsafe_allow_html=True
    )

    df_k = df_global.dropna(subset=["Waktu Respon"])
    df_k = df_k[df_k["Konselor"] != "Tidak Terjawab"]

    if df_k["Konselor"].nunique() < 2:
        st.warning("Minimal perlu 2 konselor dengan data respon.")
        st.stop()

    # ===============================
    # UJI KRUSKAL–WALLIS
    # ===============================
    groups = df_k.groupby("Konselor")["Waktu Respon"].apply(list)
    stat, p = kruskal(*groups)

    st.markdown("### 📊 Hasil Uji Kruskal–Wallis")
    st.write(f"**Statistik H:** {stat:.2f}")
    st.write(f"**p-value:** {p:.5g}")

    # ===============================
    # UJI LANJUTAN DUNN
    # ===============================
    st.markdown("### 📋 Uji Lanjutan (Dunn Post-hoc)")

    posthoc = sp.posthoc_dunn(
        df_k,
        val_col="Waktu Respon",
        group_col="Konselor",
        p_adjust="bonferroni"
    )

    st.dataframe(posthoc)

    # ===============================
    # AMBIL PASANGAN SIGNIFIKAN
    # ===============================
    alpha = 0.05
    signif_pairs = []

    for i in posthoc.index:
        for j in posthoc.columns:
            if i != j and posthoc.loc[i, j] < alpha:
                signif_pairs.append(tuple(sorted([i, j])))

    signif_pairs = sorted(set(signif_pairs))

    # ===============================
    # INTERPRETASI OTOMATIS
    # ===============================
    st.markdown("---")
    st.subheader("📝 Interpretasi")

    # Interpretasi Kruskal–Wallis
    if p < alpha:
        st.markdown(
            f"Uji **Kruskal–Wallis** menunjukkan adanya **perbedaan kinerja yang signifikan** "
            f"antar konselor medis (H = {stat:.2f}; p-value = {p:.5g})."
        )
    else:
        st.markdown(
            f"Uji **Kruskal–Wallis** menunjukkan **tidak terdapat perbedaan kinerja yang signifikan** "
            f"antar konselor medis (H = {stat:.2f}; p-value = {p:.5g})."
        )

    # Interpretasi Post-hoc
    if signif_pairs:
        st.markdown(
            "Hasil uji lanjutan (**Dunn Post-hoc**) menunjukkan bahwa "
            "**beberapa pasangan konselor memiliki perbedaan kinerja yang bermakna "
            "secara statistik (p < 0,05)**, yaitu:"
        )
        for a, b in signif_pairs:
            st.markdown(f"- **{a}** dan **{b}**")
    else:
        st.markdown(
            "Hasil uji lanjutan (**Dunn Post-hoc**) menunjukkan bahwa "
            "**tidak terdapat pasangan konselor dengan perbedaan kinerja yang bermakna "
            "secara statistik (p ≥ 0,05)**."
        )

    st.markdown(
        "📌 Temuan ini mengindikasikan adanya variasi performa antar konselor "
        "yang perlu menjadi perhatian dalam evaluasi dan pengembangan layanan."
    )
