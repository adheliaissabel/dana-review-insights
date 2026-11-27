"""Dasbor Streamlit: Analisis Tren Ulasan DANA untuk Product Manager.

Jalankan dari root repo:  streamlit run app/app.py

Sumber data (urutan prioritas)
1. data/processed/reviews_topics_sentiment.csv   (hasil NLP Orang 2, satu file lengkap)
     kolom dipakai: review_id, text_raw, text_clean, rating, date, app_version, versi_grp,
     thumbs_up, is_spike, topic_label, topic_prob, low_conf, sentimen
   + reports/ringkasan_per_topik.md (opsional, ringkasan per topik)
2. data/clean/reviews_clean.csv + data/results/reviews_topics.csv (format lama)
3. data/clean/reviews_clean.csv saja -> MODE DUMMY (topik kata kunci, sentimen dari rating)
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "reviews_topics_sentiment.csv"
SUMMARY_MD = ROOT / "reports" / "ringkasan_per_topik.md"
CLEAN_CSV = ROOT / "data" / "clean" / "reviews_clean.csv"
TOPICS_CSV = ROOT / "data" / "results" / "reviews_topics.csv"
SUMMARY_CSV = ROOT / "data" / "results" / "topic_summary.csv"

SENT_ORDER = ["negatif", "netral", "positif"]
SENT_COLOR = {"negatif": "#F2725E", "netral": "#CBD5E1", "positif": "#4F8FF7"}
UNKNOWN_VER = "tidak diketahui"
OLD_VER = "versi lama"
MIN_VER = 30  # versi dengan ulasan < MIN_VER digabung menjadi OLD_VER
ACRONYMS = {"ui": "UI", "ux": "UX", "qris": "QRIS", "otp": "OTP", "kyc": "KYC", "id": "ID"}

# Aturan kata kunci HANYA untuk data dummy (diganti hasil LDA Orang 2).
DUMMY_TOPICS = {
    "Gangguan Jaringan/Koneksi": {"jaringan", "koneksi", "internet", "sinyal", "terputus", "gangguan"},
    "Top-up & Saldo": {"saldo", "topup", "hilang", "uang", "isi"},
    "Login & Akun": {"login", "masuk", "akun", "otp", "verifikasi", "buka"},
    "DANA Cicil & Limit": {"cicil", "limit", "paylater", "tagihan"},
    "Promo & Event": {"event", "hadiah", "promo", "cashback", "voucher"},
    "Transaksi & Pembayaran": {"transaksi", "bayar", "pembayaran", "transfer", "kirim"},
    "Error, Bug & Update": {"error", "bug", "update", "lama", "lambat", "loading", "perbaiki", "crash", "macet"},
    "Pujian & Kemudahan": {"bagus", "mudah", "membantu", "cepat", "lancar", "aman", "baik"},
}
OTHER = "Lainnya"

# Catatan keterbatasan dari evaluasi Orang 2 (reports/evaluasi_nlp.md, n = 100 label manual).
# Perbarui angka ini bila evaluasi final berubah.
LIMITS = [
    "Topik cocok label manual hanya ±49% (±54% untuk ulasan yang yakin): topik adalah perkiraan, "
    "bukan fakta. Paling konsisten: Pujian Umum.",
    "Sentimen: akurasi ±72%, macro-F1 ±0,68. Kelas netral lemah, jadi porsi negatif cenderung terlalu "
    "tinggi. Selisih kecil antar kelompok (beberapa poin persen) jangan ditafsirkan sebagai perbedaan nyata.",
    "Lonjakan 6 Nov (±33% data) mendongkrak topik gangguan jaringan dan versi 2.151.0. "
    "Gunakan filter 'Tanpa hari lonjakan' untuk membandingkan.",
    "Data hanya ±17 hari: tren jangka panjang tidak dapat disimpulkan. Perbandingan versi bersifat indikasi.",
    "Rating bukan sentimen: ada keluhan yang diberi bintang 5.",
]

# Kata umum yang disembunyikan dari daftar kata kunci agar kata topik lebih terlihat.
WC_STOP = {
    "tidak", "mau", "padahal", "malah", "sering", "banyak", "sama", "semua",
    "sekarang", "selalu", "jelas", "lebih", "hari", "pakai", "belum", "jangan",
    "bikin", "makin", "kasih", "baru", "tiba", "masa", "orang", "gua",
}

st.set_page_config(page_title="DANA Review Insights", layout="wide")

def is_dark():
    """Deteksi tema aktif (Streamlit >= 1.46). Bila tidak tersedia, anggap light."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()

# Warna kartu per tema: light = kartu putih + bayangan abu, dark = kartu gelap + bayangan pekat.
if DARK:
    BG, CARD = "#0F172A", "#1E293B"
    SHADOW = "0 4px 16px rgba(0, 0, 0, 0.45)"
    TAB_OFF = "#273449"
else:
    BG, CARD = "#F3F5F9", "#FFFFFF"
    SHADOW = "0 4px 14px rgba(17, 24, 39, 0.10)"
    TAB_OFF = "#FFFFFF"

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", Arial, sans-serif; }
.stApp { background: %(bg)s; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1 { font-weight: 700; letter-spacing: -0.5px; font-size: 2rem; }

[data-testid="stSidebarHeader"] { height: 1rem; }
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }

/* kartu: putih (atau gelap di dark mode) dengan bayangan timbul */
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stExpander"] details {
    background: %(card)s;
    border: none;
    border-radius: 20px;
    box-shadow: %(shadow)s;
}
[data-testid="stMetric"] { padding: 20px 22px; }
[data-testid="stPlotlyChart"] { padding: 0; overflow: hidden; }
[data-testid="stExpander"] details { border-radius: 16px; }
[data-testid="stMetricValue"] { font-size: 2.3rem; font-weight: 700; }

/* tab pil: aktif memakai warna aksen */
div[data-baseweb="tab-list"] { gap: 8px; }
button[data-baseweb="tab"] {
    background: %(tab_off)s; border-radius: 999px; padding: 6px 20px; height: auto;
    box-shadow: %(shadow)s;
}
button[data-baseweb="tab"] p { font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { background: #4F8FF7; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
</style>
""" % dict(bg=BG, card=CARD, shadow=SHADOW, tab_off=TAB_OFF), unsafe_allow_html=True)

# Palet grafik: biru dan lime sebagai aksen, tetap terbaca di light dan dark.
PALETTE = ["#4F8FF7", "#D4EC4A", "#7FB3E6", "#2A9D8F", "#8E6BBF", "#F2A65A", "#9CA3AF", "#F2725E"]
px.defaults.template = "plotly_dark" if DARK else "plotly_white"
px.defaults.color_discrete_sequence = PALETTE


def show(target, fig):
    """Tampilkan grafik dengan gaya seragam; latar transparan agar kartu CSS yang terlihat."""
    fig.update_layout(margin=dict(l=24, r=24, t=56, b=18), title_font_size=16,
                      font_family="Segoe UI", colorway=PALETTE,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      modebar=dict(bgcolor="rgba(0,0,0,0)", color="#94A3B8", activecolor="#4F8FF7"))
    try:
        fig.update_traces(marker_cornerradius=8, selector=dict(type="bar"))
    except Exception:
        pass
    target.plotly_chart(fig, theme=None, config={"displayModeBar": "hover", "displaylogo": False,
                                                    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]})


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Memuat data...")
def load_data():
    if PROCESSED_CSV.exists():  # hasil NLP lengkap dari Orang 2
        df = pd.read_csv(PROCESSED_CSV, parse_dates=["date"])
        df["app_version_raw"] = df["app_version"].fillna(UNKNOWN_VER)
        grp = df["versi_grp"] if "versi_grp" in df.columns else df["app_version_raw"]
        df["app_version"] = grp.replace({"kosong": UNKNOWN_VER})
        df["day"] = df["date"].dt.normalize()
        df["topic_label"] = df["topic_label"].map(pretty_label, na_action="ignore").fillna(OTHER)
        df["topic_prob"] = df["topic_prob"].fillna(0.0)
        df["sentiment"] = df["sentimen"]
        df["low_conf"] = df["low_conf"].fillna(False).astype(bool) if "low_conf" in df.columns else False
        return df, True, load_summary(df["topic_label"].unique())

    if not CLEAN_CSV.exists():
        return None, False, None
    df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
    df["app_version_raw"] = df["app_version"].fillna(UNKNOWN_VER)
    ver_n = df["app_version_raw"].map(df["app_version_raw"].value_counts())
    df["app_version"] = df["app_version_raw"].where(
        (ver_n >= MIN_VER) | (df["app_version_raw"] == UNKNOWN_VER), OLD_VER)
    df["day"] = df["date"].dt.normalize()
    df["low_conf"] = False

    real = TOPICS_CSV.exists()
    if real:
        res = pd.read_csv(TOPICS_CSV)
        keep = [c for c in ["review_id", "topic_label", "topic_prob", "sentiment"] if c in res.columns]
        df = df.merge(res[keep], on="review_id", how="left")
        df["topic_label"] = df["topic_label"].map(pretty_label, na_action="ignore").fillna(OTHER)
        if "sentiment" not in df.columns:
            df["sentiment"] = df["rating"].map(rating_to_sent)
        if "topic_prob" not in df.columns:
            df["topic_prob"] = 1.0
    else:
        df["topic_label"], df["topic_prob"] = zip(*df["text_clean"].map(dummy_topic))
        df["sentiment"] = df["rating"].map(rating_to_sent)

    summ = pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else None
    if summ is not None:
        summ["topic_label"] = summ["topic_label"].map(pretty_label)
    return df, real, summ


def load_summary(labels):
    """Ambil ringkasan per topik dari reports/ringkasan_per_topik.md (judul = nama topik)."""
    if SUMMARY_MD.exists():
        text = SUMMARY_MD.read_text(encoding="utf-8")
        out = {}
        for sec in re.split(r"(?m)^#{1,6}\s*", text)[1:]:
            head, _, body = sec.partition("\n")
            for lab in sorted(labels, key=len, reverse=True):
                if lab.lower() in head.lower() and lab not in out:
                    out[lab] = body.strip()
                    break
        if out:
            return pd.DataFrame({"topic_label": list(out), "summary": list(out.values())})
    if SUMMARY_CSV.exists():
        summ = pd.read_csv(SUMMARY_CSV)
        summ["topic_label"] = summ["topic_label"].map(pretty_label)
        return summ
    return None


def pretty_label(label):
    """'gangguan_jaringan' -> 'Gangguan Jaringan'. Label yang sudah rapi dibiarkan."""
    label = str(label).strip()
    if "_" not in label and not label.islower():
        return label
    words = label.replace("_", " ").split()
    return " ".join(ACRONYMS.get(w.lower(), w.capitalize()) for w in words)


def rating_to_sent(r):
    return "negatif" if r <= 2 else "netral" if r == 3 else "positif"


def dummy_topic(text):
    tokens = set(str(text).split())
    hits = {t: len(tokens & kw) for t, kw in DUMMY_TOPICS.items()}
    total = sum(hits.values())
    if total == 0:
        return OTHER, 0.0
    best = max(hits, key=hits.get)
    return best, hits[best] / total


@st.cache_data
def top_words(texts: tuple, n: int = 80):
    c = Counter(w for t in texts for w in t.split() if w not in WC_STOP and len(w) > 2)
    return dict(c.most_common(n))


# ---------------------------------------------------------------- load
df_all, USING_REAL, summary_df = load_data()
if df_all is None:
    st.error("Data tidak ditemukan. Jalankan dari root repo dan pastikan "
             "data/processed/reviews_topics_sentiment.csv atau data/clean/reviews_clean.csv ada (git pull).")
    st.stop()

st.title("DANA Review Insights")

if not USING_REAL:
    st.warning(
        "**Mode data dummy.** Topik memakai aturan kata kunci sederhana dan sentimen diturunkan "
        "dari rating, jadi belum mewakili hasil model. Letakkan hasil Orang 2 di "
        "`data/processed/reviews_topics_sentiment.csv`, lalu muat ulang halaman.",
    )

# ---------------------------------------------------------------- filters
st.sidebar.header("Filter")
dmin, dmax = df_all["day"].min().date(), df_all["day"].max().date()
dr = st.sidebar.date_input("Rentang tanggal", (dmin, dmax), min_value=dmin, max_value=dmax)
d0, d1 = (dr[0], dr[1]) if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax)

versions = df_all["app_version"].value_counts().index.tolist()
sel_ver = st.sidebar.multiselect("Versi aplikasi", versions, default=[],
                                 placeholder="Semua versi (urut jumlah ulasan)")
r0, r1 = st.sidebar.slider("Rating", 1, 5, (1, 5))
sel_sent = st.sidebar.multiselect("Sentimen", SENT_ORDER, default=[], placeholder="Semua sentimen")
topics_all = sorted(df_all["topic_label"].unique())
sel_topic = st.sidebar.multiselect("Topik", topics_all, default=[], placeholder="Semua topik")

no_spike = st.sidebar.checkbox("Tanpa hari lonjakan (6 Nov)", value=False,
                               help="Menyingkirkan hari lonjakan agar pola hari biasa dan perbandingan versi tidak bias.")
hide_low = False
if df_all["low_conf"].any():
    hide_low = st.sidebar.checkbox("Sembunyikan ulasan topik kurang yakin", value=False,
                                   help="Ulasan sangat pendek yang topiknya kurang yakin (low_conf).")

df = df_all[(df_all["day"].dt.date >= d0) & (df_all["day"].dt.date <= d1) & df_all["rating"].between(r0, r1)]
if sel_ver:
    df = df[df["app_version"].isin(sel_ver)]
if sel_sent:
    df = df[df["sentiment"].isin(sel_sent)]
if sel_topic:
    df = df[df["topic_label"].isin(sel_topic)]
if no_spike:
    df = df[~df["is_spike"]]
if hide_low:
    df = df[~df["low_conf"]]

if df.empty:
    st.info("Tidak ada ulasan yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

# ---------------------------------------------------------------- KPI
n_days = max((d1 - d0).days + 1, 1)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total ulasan", f"{len(df):,}".replace(",", "."))
k2.metric("Rata-rata rating", f"{df['rating'].mean():.2f}")
k3.metric("Sentimen negatif", f"{(df['sentiment'] == 'negatif').mean() * 100:.1f}%")
k4.metric("Ulasan per hari (rata-rata)", f"{len(df) / n_days:,.0f}".replace(",", "."))

# ---------------------------------------------------------------- lonjakan (untuk anotasi grafik)
daily_all = df_all.groupby("day").size()
spikes = daily_all[daily_all >= 3 * daily_all.median()]

# ---------------------------------------------------------------- tabs
tab_ov, tab_topic, tab_ver, tab_exec, tab_data = st.tabs(
    ["Ringkasan", "Topik & Sentimen", "Versi Aplikasi", "Ringkasan Eksekutif", "Data"]
)

with tab_ov:
    c1, c2 = st.columns([2, 1])
    daily = df.groupby("day").agg(ulasan=("rating", "size"), rating=("rating", "mean")).reset_index()
    fig = px.bar(daily, x="day", y="ulasan", title="Ulasan per hari", labels={"day": "", "ulasan": "Jumlah ulasan"})
    for day in spikes.index:
        row = daily[daily["day"] == day]
        if not row.empty:
            fig.add_annotation(x=day, y=int(row["ulasan"].iloc[0]), text="Lonjakan", showarrow=True,
                               arrowhead=2, ay=-30, font=dict(size=12))
    show(c1, fig)
    rd = df["rating"].value_counts().sort_index().reset_index()
    rd.columns = ["rating", "jumlah"]
    fig = px.bar(rd, x="rating", y="jumlah", title="Distribusi rating", labels={"jumlah": "Jumlah ulasan"})
    fig.update_xaxes(dtick=1)
    show(c2, fig)
    fig = px.line(daily, x="day", y="rating", markers=True, title="Rata-rata rating per hari",
                  labels={"day": "", "rating": "Rating"})
    fig.update_yaxes(range=[1, 5])
    show(st, fig)

with tab_topic:
    c1, c2 = st.columns(2)
    tc = df["topic_label"].value_counts().reset_index()
    tc.columns = ["topik", "jumlah"]
    fig = px.bar(tc.sort_values("jumlah"), x="jumlah", y="topik", orientation="h",
                 title="Distribusi topik", labels={"jumlah": "Jumlah ulasan", "topik": ""})
    show(c1, fig)
    ts = df.groupby(["topic_label", "sentiment"]).size().reset_index(name="jumlah")
    fig = px.bar(ts, y="topic_label", x="jumlah", color="sentiment", orientation="h",
                 category_orders={"sentiment": SENT_ORDER}, color_discrete_map=SENT_COLOR,
                 title="Sentimen per topik",
                 labels={"jumlah": "Jumlah ulasan", "topic_label": "", "sentiment": "Sentimen"})
    show(c2, fig)
    td = df.groupby(["day", "topic_label"]).size().reset_index(name="jumlah")
    fig = px.bar(td, x="day", y="jumlah", color="topic_label", title="Tren topik per hari",
                 labels={"day": "", "jumlah": "Jumlah ulasan", "topic_label": "Topik"})
    show(st, fig)

with tab_ver:
    vs = df.groupby("app_version").agg(ulasan=("rating", "size"), rating=("rating", "mean"),
                                       negatif=("sentiment", lambda s: (s == "negatif").mean() * 100)).reset_index()
    vs = vs[vs["ulasan"] >= MIN_VER].sort_values("ulasan", ascending=False)
    if vs.empty:
        st.info("Tidak ada versi dengan minimal 30 ulasan pada filter ini.")
    else:
        c1, c2 = st.columns(2)
        fig = px.bar(vs, x="app_version", y="ulasan", title="Ulasan per versi",
                     labels={"app_version": "Versi", "ulasan": "Jumlah ulasan"})
        show(c1, fig)
        fig = px.bar(vs, x="app_version", y="rating", title="Rata-rata rating per versi",
                     labels={"app_version": "Versi", "rating": "Rating"})
        fig.update_yaxes(range=[1, 5])
        show(c2, fig)
        vt = df[df["app_version"].isin(vs["app_version"])].groupby(["app_version", "topic_label"]).size() \
            .reset_index(name="jumlah")
        fig = px.bar(vt, x="app_version", y="jumlah", color="topic_label", barmode="relative",
                     title="Komposisi topik per versi", labels={"app_version": "Versi", "jumlah": "Jumlah ulasan",
                                                                "topic_label": "Topik"})
        show(st, fig)

with tab_exec:
    summ_map = dict(zip(summary_df["topic_label"], summary_df["summary"])) if summary_df is not None else {}
    order = df["topic_label"].value_counts()
    for topic, cnt in order.items():
        sub = df[df["topic_label"] == topic]
        dom = sub["sentiment"].value_counts().idxmax()
        with st.expander(f"{topic}: {cnt / len(df) * 100:.1f}% ulasan ({cnt:,}), sentimen dominan {dom}"
                         .replace(",", "."), expanded=topic == order.index[0]):
            freqs = top_words(tuple(sub["text_clean"]))
            if topic in summ_map:
                st.markdown(f"**Ringkasan:** {summ_map[topic]}")
            else:
                kw = ", ".join(list(freqs)[:6])
                st.markdown(f"**Ringkasan:** _belum tersedia (menunggu hasil summarization Orang 2)._ "
                            f"Kata kunci utama: {kw}.")
            if freqs and topic in summ_map:
                st.markdown(f"**Kata kunci:** {', '.join(list(freqs)[:8])}.")
            st.markdown("**Contoh ulasan (jempol terbanyak):**")
            for _, r in sub.sort_values(["thumbs_up", "date"], ascending=False).head(3).iterrows():
                st.markdown(f"> {r['text_raw'][:300]}\n\n"
                            f"<sub>★{r['rating']} · {r['date']:%d %b %Y} · v{r['app_version_raw']} · "
                            f"jempol {r['thumbs_up']}</sub>", unsafe_allow_html=True)

with tab_data:
    cols = ["date", "rating", "app_version_raw", "topic_label", "sentiment", "thumbs_up", "text_raw"]
    st.dataframe(df[cols].sort_values("date", ascending=False), hide_index=True)
    st.download_button("Unduh data terfilter (CSV)", df[cols].to_csv(index=False).encode("utf-8"),
                       file_name="ulasan_terfilter.csv", mime="text/csv")