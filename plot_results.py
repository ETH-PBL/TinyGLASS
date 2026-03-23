"""
Generate paper figures and tables from experiment results.

Outputs:
  - results/table_mvtec.csv       : per-class + mean AUROC table for TinyGLASS
  - results/fig_contamination.pdf : AUROC vs contamination rate (MVTec + MMS)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'legend.handlelength': 1.0,
    'legend.handletextpad': 0.4,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.3,
    'pdf.fonttype': 42,
})

RESULTS_ROOT = "results"
RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
RATE_LABELS = ["0%", "5%", "10%", "20%", "30%"]

# Published GLASS (WideResNet50) numbers on MVTec-AD from the paper
GLASS_IMAGE_AUROC = 99.1
GLASS_PIXEL_AUROC = 98.3
GLASS_PARAMS_M = 68.9

TINYGLASS_PARAMS_M = 8.0  # ResNet18 backbone


# ---------------------------------------------------------------------------
# 1. MVTec Baseline Table
# ---------------------------------------------------------------------------

def load_mvtec_results():
    csv_path = os.path.join(RESULTS_ROOT, "tinyglass_mvtec", "results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found — skipping MVTec table")
        return None
    df = pd.read_csv(csv_path)
    return df


def build_mvtec_table(df):
    # Rename columns for clarity
    col_map = {
        "Row Names": "Class",
        "image_auroc": "Img AUROC",
        "image_ap": "Img AP",
        "pixel_auroc": "Pix AUROC",
        "pixel_ap": "Pix AP",
        "pixel_pro": "Pix PRO",
        "best_epoch": "Best Epoch",
    }
    df = df.rename(columns=col_map)

    # Convert float columns to percentages
    for col in ["Img AUROC", "Img AP", "Pix AUROC", "Pix AP", "Pix PRO"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)

    out_path = os.path.join(RESULTS_ROOT, "table_mvtec.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved MVTec table -> {out_path}")

    # Print summary
    mean_row = df[df["Class"] == "Mean"]
    if not mean_row.empty:
        img_auc = mean_row["Img AUROC"].values[0]
        pix_auc = mean_row["Pix AUROC"].values[0]
        print(f"\n=== TinyGLASS vs GLASS (MVTec-AD mean) ===")
        print(f"{'Model':<12} {'Backbone':<15} {'Params':>8} {'Img AUROC':>10} {'Pix AUROC':>10}")
        print(f"{'GLASS':<12} {'WideResNet50':<15} {f'{GLASS_PARAMS_M:.0f}M':>8} {GLASS_IMAGE_AUROC:>10.1f} {GLASS_PIXEL_AUROC:>10.1f}")
        print(f"{'TinyGLASS':<12} {'ResNet18':<15} {f'{TINYGLASS_PARAMS_M:.0f}M':>8} {img_auc:>10.1f} {pix_auc:>10.1f}")
        drop = GLASS_IMAGE_AUROC - img_auc
        compression = GLASS_PARAMS_M / TINYGLASS_PARAMS_M
        print(f"\nCompression: {compression:.1f}x  |  AUROC drop: {drop:.1f}pp")

    return df


# ---------------------------------------------------------------------------
# 2. Contamination Robustness Plot
# ---------------------------------------------------------------------------

def load_contamination_mean(dataset, rate, metric="image_auroc"):
    """Return mean metric for a given dataset and contamination rate."""
    # Try multiple float format variants to match folder naming (e.g. rate_0.1 vs rate_0.10)
    for fmt in [f"rate_{rate}", f"rate_{rate:.2f}"]:
        csv_path = os.path.join(RESULTS_ROOT, "contamination", dataset, fmt, "results.csv")
        if os.path.exists(csv_path):
            break
    else:
        return None
    df = pd.read_csv(csv_path)
    mean_row = df[df["Row Names"] == "Mean"] if "Row Names" in df.columns else df.tail(1)
    if mean_row.empty:
        return None
    val = mean_row[metric].values[0]
    return round(float(val) * 100, 2)


def build_contamination_plot():
    # (dataset_key, label, marker, color, show_in_pixel_panel)
    mvtec_color = "#6baed6"   # lighter blue for TinyGLASS on MVTec-AD
    mms_color   = "#ff7f0e"
    dataset_cfg = [
        ("mvtec", "MVTec-AD (carpet)", "o", mvtec_color, True),
        ("mms",   "MMS",               "s", mms_color,   False),
    ]
    metrics = [
        ("image_auroc", "Image AUROC (%)", GLASS_IMAGE_AUROC, "GLASS (WRN50)"),
        ("pixel_auroc", "Pixel AUROC (%)", GLASS_PIXEL_AUROC, "GLASS (WRN50)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 5.5), sharey=False)

    x = list(range(len(RATES)))  # evenly spaced: 0,1,2,3,4

    for ax, (metric, ylabel, glass_ref, glass_label) in zip(axes, metrics):
        any_data = False

        is_pixel = metric == "pixel_auroc"
        for dataset, label, marker, color, show_pixel in dataset_cfg:
            if is_pixel and not show_pixel:
                continue
            vals = [load_contamination_mean(dataset, r, metric) for r in RATES]
            if all(v is None for v in vals):
                continue
            any_data = True
            y = [v if v is not None else float("nan") for v in vals]
            base = y[0] if not pd.isna(y[0]) else None

            # Line — plot only valid points so gaps (missing rates) are bridged
            valid_x = [xi for xi, yi in zip(x, y) if not pd.isna(yi)]
            valid_y = [yi for yi in y if not pd.isna(yi)]
            ax.plot(valid_x, valid_y, color=color, linewidth=2, zorder=3, alpha=0.35)

            # Shaded degradation area — use valid points so shade follows the bridged line
            if base is not None:
                ax.fill_between(valid_x, valid_y, base, color=color, alpha=0.08)

            # Diamond markers — label here so legend shows a diamond, not a line
            ax.scatter(valid_x, valid_y, marker="D", s=28, color=color,
                       edgecolors="white", linewidths=0.5, zorder=5, label=label)

            # Value labels — above point only, no drop text
            for xi, yi in zip(x, y):
                if not pd.isna(yi):
                    ax.text(xi, yi + 0.25, f"{yi:.1f}", ha="center", va="bottom",
                            fontsize=12, color=color, fontweight="semibold")

        if not any_data:
            ax.set_visible(False)
            continue

        # GLASS reference point at 0% contamination only (published clean result)
        # Use the same color as MVTec-AD since GLASS is also evaluated on MVTec
        ax.scatter([0], [glass_ref], marker="*", s=90, color=mvtec_color,
                   zorder=6, label=f"{glass_label}")
        ax.annotate(f"{glass_ref:.1f}", (0, glass_ref), textcoords="offset points",
                    xytext=(9, -11), fontsize=12, color=mvtec_color, fontweight="semibold",
                    arrowprops=dict(arrowstyle="-", color=mvtec_color, lw=0.7))

        # Alternating column shading — one band per contamination rate
        for i in x:
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)

        # Thin vertical separator at each rate
        for i in x:
            ax.axvline(i, color="#dddddd", linewidth=0.8, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(RATE_LABELS)
        ax.set_xlabel("Contamination Rate", labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(80, 100)
        ax.set_yticks([80, 85, 90, 95, 100])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Booktabs-style legend table drawn manually (toprule / midrule / bottomrule)
    table_rows = [
        ("◆", mvtec_color, "TinyGLASS (ResNet-18)", "MVTec-AD (carpet)"),
        ("◆", mms_color,   "TinyGLASS (ResNet-18)", "MMS"),
        ("★", mvtec_color, "GLASS (WRN50)",          "MVTec-AD"),
    ]

    plt.tight_layout(pad=0.5, w_pad=0.8)
    plt.subplots_adjust(top=0.76, bottom=0.10)

    tbl_ax = fig.add_axes([0.02, 0.78, 0.96, 0.21])
    tbl_ax.set_xlim(0, 1)
    tbl_ax.set_ylim(0, 1)
    tbl_ax.set_axis_off()

    n_rows  = len(table_rows) + 1          # header + data rows
    rh      = 1.0 / n_rows                 # row height in axes fraction
    fs      = 13                           # body font size
    fs_sym  = 15                           # symbol font size
    # Column left-edge x positions (axes fraction)
    col_x   = [0.00, 0.07, 0.58]
    col_hdr = ["",   "Model", "Dataset"]

    # Horizontal rules
    lw_thick = 1.5
    lw_thin  = 0.8
    tbl_ax.axhline(1.0,        color="black", linewidth=lw_thick)   # toprule
    tbl_ax.axhline(1.0 - rh,   color="black", linewidth=lw_thin)    # midrule
    tbl_ax.axhline(0.0,        color="black", linewidth=lw_thick)   # bottomrule

    # Header row
    y_hdr = 1.0 - rh / 2
    for x, label in zip(col_x[1:], col_hdr[1:]):
        tbl_ax.text(x + 0.01, y_hdr, label, ha="left", va="center",
                    fontsize=fs, fontweight="bold")

    # Data rows
    for i, (sym, color, model, dataset) in enumerate(table_rows):
        y = 1.0 - (i + 1.5) * rh
        tbl_ax.text(col_x[0] + 0.025, y, sym,     ha="center", va="center",
                    fontsize=fs_sym, color=color)
        tbl_ax.text(col_x[1] + 0.01,  y, model,   ha="left",   va="center",
                    fontsize=fs)
        tbl_ax.text(col_x[2] + 0.01,  y, dataset, ha="left",   va="center",
                    fontsize=fs)

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    out_path = os.path.join(RESULTS_ROOT, "fig_contamination.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved contamination plot -> {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=== MVTec Table ===")
    df_mvtec = load_mvtec_results()
    if df_mvtec is not None:
        build_mvtec_table(df_mvtec)

    print("\n=== Contamination Plot ===")
    build_contamination_plot()
