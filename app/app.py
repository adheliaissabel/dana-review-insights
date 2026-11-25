"""Dasbor Streamlit: Analisis Tren Ulasan DANA untuk Product Manager.

Jalankan dari root repo:  streamlit run app/app.py

Sumber data
- data/clean/reviews_clean.csv            (Orang 1, WAJIB)
- data/results/reviews_topics.csv         (Orang 2, opsional)
    kolom: review_id, topic_label, topic_prob, sentiment (positif/netral/negatif)
- data/results/topic_summary.csv          (Orang 2, opsional)
    kolom: topic_label, summary

Jika file Orang 2 belum ada, dasbor memakai DATA DUMMY (topik berbasis kata kunci
sederhana + sentimen dari rating) dan menampilkan banner peringatan. Saat file asli
diletakkan di data/results/, dasbor otomatis memakainya tanpa mengubah kode.
"""
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "clean" / "reviews_clean.csv"
TOPICS_CSV = ROOT / "data" / "results" / "reviews_topics.csv"
SUMMARY_CSV = ROOT / "data" / "results" / "topic_summary.csv"

SENT_ORDER = ["negatif", "netral", "positif"]
SENT_COLOR = {"negatif": "#d6604d", "netral": "#a0a0a0", "positif": "#4393c3"}
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

# Kata umum yang disembunyikan hanya di word cloud agar kata topik lebih terlihat.
WC_STOP = {
    "tidak", "mau", "padahal", "malah", "sering", "banyak", "sama", "semua",
    "sekarang", "selalu", "jelas", "lebih", "hari", "pakai", "belum", "jangan",
    "bikin", "makin", "kasih", "baru", "tiba", "masa", "orang", "gua",
}

st.set_page_config(page_title="DANA Review Insights", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Memuat data...")
def load_data():
    if not CLEAN_CSV.exists():
        return None, False, None
    df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
    df["app_version_raw"] = df["app_version"].fillna(UNKNOWN_VER)
    ver_n = df["app_version_raw"].map(df["app_version_raw"].value_counts())
    df["app_version"] = df["app_version_raw"].where(
        (ver_n >= MIN_VER) | (df["app_version_raw"] == UNKNOWN_VER), OLD_VER)
    df["day"] = df["date"].dt.normalize()

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


@st.cache_data
def wordcloud_image(freqs: tuple):
    from wordcloud import WordCloud

    wc = WordCloud(width=800, height=320, background_color="white", colormap="tab10")
    return wc.generate_from_frequencies(dict(freqs)).to_array()


# ---------------------------------------------------------------- load
df_all, USING_REAL, summary_df = load_data()
if df_all is None:
    st.error(f"File tidak ditemukan: {CLEAN_CSV}. Jalankan dari root repo dan pastikan "
             "data/clean/reviews_clean.csv sudah ada (git pull).")
    st.stop()

st.title("DANA Review Insights")
st.caption("Tren ulasan Google Play untuk Product Manager: topik keluhan, sentimen, dan lonjakan.")

if not USING_REAL:
    st.warning(
        "**Mode data dummy.** Topik memakai aturan kata kunci sederhana dan sentimen diturunkan "
        "dari rating, jadi belum mewakili hasil model. Letakkan hasil Orang 2 di "
        "`data/results/reviews_topics.csv` dan `topic_summary.csv`, lalu muat ulang halaman.",
        icon="⚠️",
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

df = df_all[(df_all["day"].dt.date >= d0) & (df_all["day"].dt.date <= d1) & df_all["rating"].between(r0, r1)]
if sel_ver:
    df = df[df["app_version"].isin(sel_ver)]
if sel_sent:
    df = df[df["sentiment"].isin(sel_sent)]
if sel_topic:
    df = df[df["topic_label"].isin(sel_topic)]

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

# ---------------------------------------------------------------- alerts (dihitung dari seluruh data)
daily_all = df_all.groupby("day").size()
base = daily_all.median()
spikes = daily_all[daily_all >= 3 * base]
for day, cnt in spikes.items():
    sub = df_all[df_all["day"] == day]
    rest = df_all[df_all["day"] != day]
    top_t = sub["topic_label"].value_counts().head(2)
    top_txt = ", ".join(f"{t} ({v / len(sub) * 100:.0f}%)" for t, v in top_t.items())
    cnt_s = f"{cnt:,}".replace(",", ".")
    st.error(
        f"**Lonjakan ulasan {day:%d %b %Y}:** {cnt_s} ulasan (≈{cnt / base:.0f}x median harian {base:.0f}), "
        f"rating rata-rata {sub['rating'].mean():.2f} vs {rest['rating'].mean():.2f} pada hari lain. "
        f"Topik dominan: {top_txt}.",
        icon="🚨",
    )

# ---------------------------------------------------------------- tabs
tab_ov, tab_topic, tab_ver, tab_exec, tab_data = st.tabs(
    ["Ringkasan", "Topik & Sentimen", "Versi Aplikasi", "Ringkasan Eksekutif", "Data"]
)

with tab_ov:
    c1, c2 = st.columns([2, 1])
    daily = df.groupby("day").agg(ulasan=("rating", "size"), rating=("rating", "mean")).reset_index()
    fig = px.bar(daily, x="day", y="ulasan", title="Ulasan per hari", labels={"day": "", "ulasan": "Jumlah ulasan"})
    c1.plotly_chart(fig)
    rd = df["rating"].value_counts().sort_index().reset_index()
    rd.columns = ["rating", "jumlah"]
    fig = px.bar(rd, x="rating", y="jumlah", title="Distribusi rating", labels={"jumlah": "Jumlah ulasan"})
    fig.update_xaxes(dtick=1)
    c2.plotly_chart(fig)
    fig = px.line(daily, x="day", y="rating", markers=True, title="Rata-rata rating per hari",
                  labels={"day": "", "rating": "Rating"})
    fig.update_yaxes(range=[1, 5])
    st.plotly_chart(fig)

with tab_topic:
    c1, c2 = st.columns(2)
    tc = df["topic_label"].value_counts().reset_index()
    tc.columns = ["topik", "jumlah"]
    fig = px.bar(tc.sort_values("jumlah"), x="jumlah", y="topik", orientation="h",
                 title="Distribusi topik", labels={"jumlah": "Jumlah ulasan", "topik": ""})
    c1.plotly_chart(fig)
    ts = df.groupby(["topic_label", "sentiment"]).size().reset_index(name="jumlah")
    fig = px.bar(ts, y="topic_label", x="jumlah", color="sentiment", orientation="h",
                 category_orders={"sentiment": SENT_ORDER}, color_discrete_map=SENT_COLOR,
                 title="Sentimen per topik",
                 labels={"jumlah": "Jumlah ulasan", "topic_label": "", "sentiment": "Sentimen"})
    c2.plotly_chart(fig)
    td = df.groupby(["day", "topic_label"]).size().reset_index(name="jumlah")
    fig = px.bar(td, x="day", y="jumlah", color="topic_label", title="Tren topik per hari",
                 labels={"day": "", "jumlah": "Jumlah ulasan", "topic_label": "Topik"})
    st.plotly_chart(fig)

with tab_ver:
    st.caption(f"Versi dengan kurang dari {MIN_VER} ulasan digabung menjadi '{OLD_VER}' agar rata-rata tidak "
               f"menyesatkan; ulasan tanpa info versi dikelompokkan sebagai '{UNKNOWN_VER}'.")
    vs = df.groupby("app_version").agg(ulasan=("rating", "size"), rating=("rating", "mean"),
                                       negatif=("sentiment", lambda s: (s == "negatif").mean() * 100)).reset_index()
    vs = vs[vs["ulasan"] >= MIN_VER].sort_values("ulasan", ascending=False)
    if vs.empty:
        st.info("Tidak ada versi dengan minimal 30 ulasan pada filter ini.")
    else:
        c1, c2 = st.columns(2)
        fig = px.bar(vs, x="app_version", y="ulasan", title="Ulasan per versi",
                     labels={"app_version": "Versi", "ulasan": "Jumlah ulasan"})
        c1.plotly_chart(fig)
        fig = px.bar(vs, x="app_version", y="rating", title="Rata-rata rating per versi",
                     labels={"app_version": "Versi", "rating": "Rating"})
        fig.update_yaxes(range=[1, 5])
        c2.plotly_chart(fig)
        vt = df[df["app_version"].isin(vs["app_version"])].groupby(["app_version", "topic_label"]).size() \
            .reset_index(name="jumlah")
        fig = px.bar(vt, x="app_version", y="jumlah", color="topic_label", barmode="relative",
                     title="Komposisi topik per versi", labels={"app_version": "Versi", "jumlah": "Jumlah ulasan",
                                                                "topic_label": "Topik"})
        st.plotly_chart(fig)

with tab_exec:
    st.caption("Per topik: porsi ulasan, sentimen dominan, ringkasan, kata kunci, dan contoh ulasan asli.")
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
            if freqs:
                try:
                    st.image(wordcloud_image(tuple(freqs.items())), width=700)
                except Exception:
                    st.bar_chart(pd.Series(freqs).head(15))
            st.markdown("**Contoh ulasan (jempol terbanyak):**")
            for _, r in sub.sort_values(["thumbs_up", "date"], ascending=False).head(3).iterrows():
                st.markdown(f"> {r['text_raw'][:300]}\n\n"
                            f"<sub>★{r['rating']} · {r['date']:%d %b %Y} · v{r['app_version_raw']} · "
                            f"👍 {r['thumbs_up']}</sub>", unsafe_allow_html=True)

with tab_data:
    cols = ["date", "rating", "app_version_raw", "topic_label", "sentiment", "thumbs_up", "text_raw"]
    st.dataframe(df[cols].sort_values("date", ascending=False), hide_index=True)
    st.download_button("Unduh data terfilter (CSV)", df[cols].to_csv(index=False).encode("utf-8"),
                       file_name="ulasan_terfilter.csv", mime="text/csv")
