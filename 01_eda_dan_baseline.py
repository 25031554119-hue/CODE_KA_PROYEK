import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))  # folder tempat skrip ini berada

NAVY, TEAL, AMBER, GREY = "#1B2A49", "#1C7C7D", "#E8963A", "#8A94A6"
COLORS = {"Dropout": AMBER, "Enrolled": GREY, "Graduate": TEAL}


def load_data(path):
    if not os.path.exists(path):
        raise SystemExit(f"File data tidak ditemukan: {path}\nLetakkan data.csv di folder yang sama dengan skrip ini, atau jalankan dengan --data <lokasi file>.")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # rapikan nama kolom (ada spasi/tab tersembunyi dan typo bawaan dataset)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Nacionality": "Nationality"})
    return df


def check_dataset(df):
    print("=== RINGKASAN DATASET ===")
    print(f"Jumlah baris : {df.shape[0]}  (syarat >= 1000)")
    print(f"Jumlah kolom : {df.shape[1]}  (syarat >= 20)  -> {df.shape[1]-1} fitur + 1 target")
    print(f"Missing value: {int(df.isna().sum().sum())}")
    print(f"Duplikat     : {int(df.duplicated().sum())}")
    print("\nDistribusi target:")
    print(pd.concat([df.Target.value_counts(),
                     df.Target.value_counts(normalize=True).round(3)], axis=1))


def plot_eda(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    order = ["Graduate", "Dropout", "Enrolled"]

    # 1. distribusi target
    cnt = df.Target.value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(cnt.index, cnt.values, color=[COLORS[k] for k in cnt.index])
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v}\n({v/len(df):.1%})",
                ha="center", va="bottom", fontsize=9, color=NAVY)
    ax.set_ylim(0, cnt.max() * 1.25)
    ax.set_ylabel("Jumlah mahasiswa")
    ax.set_title("Distribusi kelas target", color=NAVY, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig1_distribusi_target.png", dpi=200); plt.close(fig)

    # 2. persentase dropout menurut faktor finansial
    d = (df.Target == "Dropout")
    items = [("Biaya kuliah\nlunas", df["Tuition fees up to date"] == 1),
             ("Biaya kuliah\nbelum lunas", df["Tuition fees up to date"] == 0),
             ("Penerima\nbeasiswa", df["Scholarship holder"] == 1),
             ("Bukan penerima\nbeasiswa", df["Scholarship holder"] == 0),
             ("Berstatus\ndebitur", df["Debtor"] == 1),
             ("Bukan\ndebitur", df["Debtor"] == 0)]
    labels = [i[0] for i in items]
    vals = [d[i[1]].mean() * 100 for i in items]
    cols = [TEAL, AMBER, TEAL, AMBER, AMBER, TEAL]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=NAVY)
    ax.axhline(d.mean() * 100, color=NAVY, ls="--", lw=1)
    ax.text(-0.45, d.mean() * 100 + 1.5, f"rata-rata {d.mean()*100:.1f}%", ha="left", fontsize=8, color=NAVY)
    ax.set_ylabel("% mahasiswa dropout"); ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Tingkat dropout menurut faktor finansial", color=NAVY, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig2_dropout_finansial.png", dpi=200); plt.close(fig)

    # 3. mata kuliah lulus semester 1 & 2 menurut target
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), sharey=True)
    for ax, col, ttl in zip(axes, ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"],
                            ["Semester 1", "Semester 2"]):
        data = [df.loc[df.Target == k, col] for k in order]
        try:
            bp = ax.boxplot(data, patch_artist=True, tick_labels=order, medianprops=dict(color=NAVY))
        except TypeError:  # matplotlib < 3.9
            bp = ax.boxplot(data, patch_artist=True, labels=order, medianprops=dict(color=NAVY))
        for patch, k in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS[k]); patch.set_alpha(0.85)
        ax.set_title(ttl, color=NAVY, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Jumlah mata kuliah lulus")
    fig.suptitle("Prestasi akademik awal menurut status akhir", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_mk_lulus.png", dpi=200); plt.close(fig)

    # 4. korelasi dengan Dropout (top 10 mutlak)
    corr = df.drop(columns="Target").corrwith(d.astype(int))
    top = corr.reindex(corr.abs().sort_values(ascending=False).index[:10])[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh([i.replace("Curricular units", "CU") for i in top.index], top.values,
            color=[AMBER if v > 0 else TEAL for v in top.values])
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_xlabel("Korelasi dengan status Dropout (1 = dropout)")
    ax.set_title("10 fitur dengan korelasi terkuat", color=NAVY, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig4_korelasi.png", dpi=200); plt.close(fig)


def baseline(df):
    """Baseline cepat untuk memastikan masalah layak dimodelkan (bukan hasil final)."""
    X, y = df.drop(columns="Target"), df.Target
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)),
    }
    try:  # XGBoost opsional: pip install xgboost
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="mlogloss", random_state=42)
    except ImportError:
        print("XGBoost belum terpasang, dilewati (pip install xgboost)")
    print("\n=== BASELINE (Stratified 5-fold CV) ===")
    for name, m in models.items():
        pred = cross_val_predict(m, X, y, cv=cv)
        print(f"\n{name}  | macro-F1 = {f1_score(y, pred, average='macro'):.3f}")
        print(classification_report(y, pred, digits=3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data.csv"))
    ap.add_argument("--figdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args()
    df = load_data(a.data)
    check_dataset(df)
    plot_eda(df, a.figdir)
    if not a.skip_baseline:
        baseline(df)
