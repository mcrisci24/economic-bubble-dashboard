"""Build supervised ML project presentation with python-pptx.

Run from the project root:
    python presentation/build_pptx.py
"""
from __future__ import annotations
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY  = RGBColor(0x1E, 0x27, 0x61)
TEAL  = RGBColor(0x1C, 0x72, 0x93)
MINT  = RGBColor(0x0D, 0x94, 0x88)
CORAL = RGBColor(0xF9, 0x61, 0x67)
ICE   = RGBColor(0xCA, 0xDC, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFF_W = RGBColor(0xF4, 0xF7, 0xFB)
DARK  = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x64, 0x74, 0x8B)

W = Inches(10)
H = Inches(5.625)


def dark_slide(sl):
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = NAVY

def light_slide(sl):
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = OFF_W

def txt(sl, text, x, y, w, h, *, size=14, bold=False, color=DARK,
        align=PP_ALIGN.LEFT, italic=False):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return tb

def rect(sl, x, y, w, h, color: RGBColor, border=False):
    shp = sl.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if border:
        shp.line.color.rgb = MUTED
        shp.line.width = Pt(0.5)
    else:
        shp.line.fill.background()
    return shp

def title_bar(sl, title, sub=None):
    rect(sl, 0.45, 0.22, 0.07, 0.65 if not sub else 0.85, TEAL)
    txt(sl, title, 0.62, 0.18, 9.0, 0.5, size=26, bold=True, color=NAVY)
    if sub:
        txt(sl, sub, 0.62, 0.65, 9.0, 0.28, size=12, color=TEAL)

def bullets(sl, items, x, y, w, h, size=13, color=DARK):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(3)
        r = p.add_run()
        r.text = "▸  " + item
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = "Calibri"

def pg(sl, n, total, dark=False):
    c = ICE if dark else MUTED
    txt(sl, f"{n}/{total}", 9.4, 5.2, 0.5, 0.3, size=10, color=c, align=PP_ALIGN.RIGHT)

def stat_card(sl, x, y, w, h, value, label, vc=MINT):
    rect(sl, x, y, w, h, WHITE, border=True)
    rect(sl, x, y, w, 0.06, vc)
    txt(sl, value, x+0.08, y+0.13, w-0.16, 0.52, size=28, bold=True, color=vc, align=PP_ALIGN.CENTER)
    txt(sl, label, x+0.05, y+0.68, w-0.1, 0.38, size=10, color=MUTED, align=PP_ALIGN.CENTER)

def callout(sl, x, y, w, h, heading, body, hc=TEAL):
    rect(sl, x, y, w, h, RGBColor(0xE0, 0xF2, 0xFE), border=True)
    txt(sl, heading, x+0.12, y+0.08, w-0.2, 0.3, size=12, bold=True, color=hc)
    txt(sl, body, x+0.12, y+0.38, w-0.2, h-0.45, size=11, color=DARK)


# ---------------------------------------------------------------------------
def build(out: Path):
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]
    TOTAL = 15

    # ---- 1. Title --------------------------------------------------------
    sl = prs.slides.add_slide(blank)
    dark_slide(sl)
    txt(sl, "Supervised Machine Learning for", 0.7, 0.6, 8.6, 0.55,
        size=22, color=ICE, align=PP_ALIGN.CENTER)
    txt(sl, "Future Drawdown Risk Classification", 0.7, 1.1, 8.6, 0.75,
        size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rect(sl, 3.5, 1.92, 3.0, 0.04, MINT)
    txt(sl, "Novel non-Kaggle dataset  ·  Imbalanced binary classification  ·  Rare-event evaluation",
        0.5, 2.15, 9.0, 0.4, size=13, color=ICE, align=PP_ALIGN.CENTER)
    txt(sl, "The interactive dashboard is the interpretation layer.  The ML pipeline is the deliverable.",
        0.8, 2.65, 8.4, 0.38, size=12, italic=True, color=RGBColor(0x8B, 0xA4, 0xD4), align=PP_ALIGN.CENTER)
    txt(sl, "Mark Crisci  ·  2026  ·  Educational research — not investment advice",
        0.5, 4.9, 9.0, 0.38, size=11, color=RGBColor(0x8B, 0xA4, 0xD4), align=PP_ALIGN.CENTER)
    pg(sl, 1, TOTAL, dark=True)

    # ---- 2. Why This Problem --------------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Why This Problem?",
              "Rare-event binary classification on financial market time series")
    pg(sl, 2, TOTAL)
    bullets(sl, [
        "Bubble episodes recur across asset classes: dot-com 2000, housing 2008, crypto 2021 — each with recognisable precursors",
        "Exact crash timing is widely regarded as unpredictable — but supervised classification of pre-drawdown RISK is a tractable ML problem",
        "Framing: does today's feature vector resemble historical pre-drawdown regimes?",
        "Evaluation challenge: ~8.5% positive-class rate globally — standard accuracy is misleading; PR-AUC, MCC, and top-k lift are required",
        "Dataset is original (Yahoo Finance + FRED), not pre-packaged — requires careful target construction and leakage prevention",
    ], 0.55, 1.2, 9.1, 3.9)
    rect(sl, 0.5, 4.88, 9.0, 0.45, RGBColor(0xE0, 0xF2, 0xFE))
    txt(sl, "Goal: build a defensible rare-event classifier; evaluate it honestly; show what the models CAN and CANNOT conclude.",
        0.65, 4.92, 8.7, 0.38, size=11, bold=True, color=NAVY)

    # ---- 3. Dataset Architecture ----------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Dataset: Original, Not Kaggle",
              "81 tickers · 597,806 weekly rows · 1927–2026 · Yahoo Finance + FRED")
    pg(sl, 3, TOTAL)
    for cx, val, lbl, vc in [
        (0.4,  "81",      "Tickers (ETFs, stocks,\ncrypto, macro)", TEAL),
        (2.75, "597,806", "Weekly rows",                            MINT),
        (5.1,  "99 yrs",  "Historical coverage",                    TEAL),
        (7.45, "4",       "Asset segments",                         MINT),
    ]:
        stat_card(sl, cx, 1.2, 2.15, 1.15, val, lbl, vc)

    seg_data = [
        ("global",               "All tickers",  "−25%"),
        ("broad_index_etf",      "SPY, QQQ …",   "−20%"),
        ("mega_cap_ai_tech",     "AAPL, NVDA …", "−35%"),
        ("speculative_high_vol", "BTC, ARKK …",  "−50%"),
    ]
    txt(sl, "Segment-specific drawdown thresholds — because −20% in SPY ≠ −20% in BTC:",
        0.55, 2.55, 9.0, 0.3, size=13, bold=True, color=NAVY)
    for i, (seg, tickers, thr) in enumerate(seg_data):
        y = 2.95 + i * 0.42
        bg = NAVY if i == 0 else (RGBColor(0xF1, 0xF5, 0xF9) if i % 2 else WHITE)
        fc = WHITE if i == 0 else DARK
        for j, (cell, cw) in enumerate(zip([seg, tickers, thr], [3.0, 3.8, 1.8])):
            cx = 0.5 + sum([3.0, 3.8, 1.8][:j])
            rect(sl, cx, y, cw, 0.35, bg)
            txt(sl, cell, cx+0.1, y+0.05, cw-0.15, 0.27, size=12, bold=(i==0), color=fc)

    # ---- 4. Target Label ------------------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Target Label: burst_6m_segment",
              "Binary: 1 if segment-adjusted major drawdown within next 26 weekly observations")
    pg(sl, 4, TOTAL)
    bullets(sl, [
        "Positive rate: 6.8% (mega_cap_ai_tech) to 11.0% (broad_index_etf) — imbalanced by design",
        "Chronological 70/15/15 split by unique date — no shuffling, no future leakage",
        "target_valid_burst_6m_segment flag excludes overlap windows (ongoing burst rows not labelled as new onset)",
        "26-week forward window: long enough to be actionable, short enough to be testable",
    ], 0.55, 1.2, 9.1, 2.2, size=14)

    txt(sl, "Class balance by scope (test split):", 0.55, 3.45, 9.0, 0.3, size=13, bold=True, color=NAVY)
    rows = [
        ("Scope",                "Positive rate", "Test rows"),
        ("global",               "8.5%",          "54,244"),
        ("broad_index_etf",      "11.0%",         "~16,000"),
        ("mega_cap_ai_tech",     "6.8%",          "9,481"),
        ("speculative_high_vol", "10.3%",         "~11,700"),
    ]
    for i, row in enumerate(rows):
        y = 3.82 + i * 0.34
        bg = NAVY if i == 0 else (RGBColor(0xF1, 0xF5, 0xF9) if i%2 else WHITE)
        fc = WHITE if i == 0 else DARK
        for j, (cell, cw) in enumerate(zip(row, [3.5, 2.5, 2.5])):
            cx = 0.5 + sum([3.5, 2.5, 2.5][:j])
            rect(sl, cx, y, cw, 0.31, bg)
            txt(sl, cell, cx+0.1, y+0.04, cw-0.15, 0.24, size=11, bold=(i==0), color=fc)

    # ---- 5. EDA & Class Imbalance --------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "EDA: Class Imbalance & Feature Distributions",
              "Why accuracy fails · What the data tells us before modeling")
    pg(sl, 5, TOTAL)
    left = [
        "8.5% positive rate — majority-class classifier gets 91.5% accuracy, 0% recall",
        "Pre-burst weeks show elevated RSI extremes, higher short-term vol, momentum divergence",
        "Valuation multiples alone are weak predictors — timing signal requires velocity features",
        "High within-ticker autocorrelation: one regime = hundreds of correlated positive rows",
    ]
    right = [
        "→ accuracy is misleading for this problem",
        "→ PR-AUC, MCC, and top-k lift are the right metrics",
        "→ feature engineering must include momentum, RSI, and drawdown velocity",
        "→ effective sample size is far smaller than 597K rows",
    ]
    txt(sl, "EDA finding", 0.55, 1.15, 4.5, 0.3, size=13, bold=True, color=NAVY)
    txt(sl, "Modeling implication", 5.2, 1.15, 4.5, 0.3, size=13, bold=True, color=TEAL)
    rect(sl, 4.88, 1.1, 0.04, 4.2, ICE)
    bullets(sl, left,  0.5, 1.45, 4.25, 3.6, size=12)
    bullets(sl, right, 5.0, 1.45, 4.65, 3.6, size=12, color=TEAL)

    # ---- 6. Feature Engineering ----------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Feature Engineering & Transformations",
              "Baseline families + Phase 8 transformed features (88 additional columns)")
    pg(sl, 6, TOTAL)
    base_fams = [
        ("Price momentum & RSI", "7/14/26-week returns, RSI, distance from 200-DMA"),
        ("Volume ratios",        "Relative volume vs 12-week avg"),
        ("Valuation",            "P/E, P/S, P/B, EV/EBITDA, market cap"),
        ("Macro",                "Yield curve spread, VIX, credit spreads"),
        ("Rule-based scores",    "Hindenburg Omen, CAPE excess, yield inversion"),
        ("Drawdown & vol",       "Rolling max drawdown 4w/13w/26w, realised vol"),
    ]
    txt(sl, "Baseline feature families:", 0.55, 1.2, 4.5, 0.3, size=13, bold=True, color=NAVY)
    for i, (name, desc) in enumerate(base_fams):
        y = 1.55 + i * 0.55
        rect(sl, 0.5, y, 0.06, 0.4, TEAL)
        txt(sl, name, 0.68, y, 2.3, 0.22, size=11, bold=True, color=NAVY)
        txt(sl, desc, 0.68, y+0.22, 4.1, 0.22, size=10, color=DARK)

    txt(sl, "Phase 8 transformed families (88 new columns):", 5.15, 1.2, 4.6, 0.3, size=13, bold=True, color=NAVY)
    tfams = [
        "log1p / signed-log compression",
        "Winsorisation (fit on TRAIN only)",
        "Rolling z-score by ticker (shift(1) guard)",
        "Expanding percentile rank",
        "Regime flags (vol / RSI thresholds)",
        "Interaction features (PE × momentum …)",
        "Vol-adjusted returns",
        "Drawdown-state binary flags",
        "Squared terms",
    ]
    for i, f in enumerate(tfams):
        y = 1.55 + i * 0.44
        rect(sl, 5.12, y, 0.06, 0.32, MINT)
        txt(sl, f, 5.3, y+0.02, 4.4, 0.28, size=11, color=DARK)

    rect(sl, 0.5, 5.12, 9.0, 0.32, RGBColor(0xE0, 0xF2, 0xFE))
    txt(sl, "Leakage prevention: winsor/regime thresholds fit on TRAIN only (TransformFitState); rolling z-score uses shift(1) lookback guard.",
        0.65, 5.16, 8.7, 0.24, size=10, color=NAVY)

    # ---- 7. Benchmark Model --------------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Benchmark: Logistic Regression (L2)",
              "Simplest defensible baseline for imbalanced binary classification")
    pg(sl, 7, TOTAL)
    for cx, val, lbl, vc in [
        (0.4,  "0.128",  "Test PR-AUC\n(base rate 0.085)", MINT),
        (2.75, "0.615",  "Test ROC-AUC",                   TEAL),
        (5.1,  "2.05×",  "Top-5% lift",                    MINT),
        (7.45, "0.077",  "MCC",                            TEAL),
    ]:
        stat_card(sl, cx, 1.25, 2.15, 1.2, val, lbl, vc)

    txt(sl, "Why Logistic Regression as benchmark?", 0.55, 2.65, 9.0, 0.3, size=14, bold=True, color=NAVY)
    bullets(sl, [
        "Regularised linear models are the standard first baseline for rare-event classification",
        "L2 penalty controls for small effective sample size (few independent bubble events)",
        "Coefficients are interpretable — feature direction and magnitude are directly readable",
        "Any more-complex model that fails to beat this baseline does not justify its added complexity",
        "Threshold tuned on validation split, not hard-coded at 0.50",
    ], 0.55, 3.0, 9.1, 2.4, size=13)

    # ---- 8. All Models Tested ------------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "All Models Tested",
              "5 baseline families + Rule-Based baseline · LightGBM in transformed-feature pipeline only")
    pg(sl, 8, TOTAL)

    model_rows = [
        # (model, role, pr_auc, note)
        ("Logistic Regression",    "Benchmark & recommended overall",      "0.128", "global"),
        ("Elastic Net Logistic",   "Regularised linear comparison",         "0.128", "global"),
        ("Random Forest",          "Nonlinear tree comparison",             "0.101", "global"),
        ("Balanced Random Forest", "Imbalance-aware tree (class_weight)",   "0.108", "global"),
        ("XGBoost",                "Boosted-tree challenger",               "0.109", "global"),
        ("Rule-Based Score",       "Interpretable sanity-check baseline",   "0.096", "global"),
        ("LightGBM",               "Phase 8 transformed-feature only",     "0.119", "broad ETF"),
    ]
    hdrs = ["Model", "Role", "Test PR-AUC", "Scope / note"]
    for j, (h, cw) in enumerate(zip(hdrs, [3.0, 3.2, 1.5, 1.9])):
        cx = 0.4 + sum([3.0, 3.2, 1.5, 1.9][:j])
        rect(sl, cx, 1.2, cw, 0.35, NAVY)
        txt(sl, h, cx+0.1, 1.24, cw-0.15, 0.27, size=12, bold=True, color=WHITE)
    for i, (m, role, auc, note) in enumerate(model_rows):
        y = 1.58 + i * 0.44
        is_lgbm = "LightGBM" in m
        bg = RGBColor(0xFE, 0xF9, 0xEE) if is_lgbm else (RGBColor(0xF1, 0xF5, 0xF9) if i%2 else WHITE)
        for j, (cell, cw) in enumerate(zip([m, role, auc, note], [3.0, 3.2, 1.5, 1.9])):
            cx = 0.4 + sum([3.0, 3.2, 1.5, 1.9][:j])
            rect(sl, cx, y, cw, 0.38, bg)
            fc = CORAL if (is_lgbm and j == 0) else DARK
            txt(sl, cell, cx+0.1, y+0.06, cw-0.15, 0.28, size=11, bold=(is_lgbm and j==0), color=fc)
    rect(sl, 0.4, 4.72, 9.2, 0.65, RGBColor(0xFE, 0xF9, 0xEE))
    txt(sl,
        "LightGBM (orange row): trained only in the transformed-feature experiment (Phase 8). "
        "It is NOT in the baseline comparison. The baseline uses Balanced RF instead. "
        "This is explicit in the dashboard and all presentation materials.",
        0.55, 4.78, 9.0, 0.52, size=11, italic=True, color=CORAL)

    # ---- 9. Splits & Metrics -------------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Train / Val / Test & Evaluation Metrics",
              "Chronological 70/15/15 split · no shuffle · threshold tuned on val")
    pg(sl, 9, TOTAL)

    for cx, w2, label, desc, vc in [
        (0.5,  2.8, "Train 70%",      "1927 – ~2007\nModel fitting", NAVY),
        (3.45, 2.8, "Validation 15%", "~2007 – ~2016\nThreshold tuning", TEAL),
        (6.4,  2.8, "Test 15%",       "~2016 – 2026\nFinal evaluation", MINT),
    ]:
        rect(sl, cx, 1.25, w2, 1.0, vc)
        txt(sl, label, cx+0.15, 1.33, w2-0.25, 0.4, size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        txt(sl, desc, cx+0.15, 1.73, w2-0.25, 0.45, size=11, color=ICE, align=PP_ALIGN.CENTER)

    txt(sl, "Evaluation metrics for imbalanced rare-event classification:", 0.55, 2.5, 9.0, 0.3, size=14, bold=True, color=NAVY)
    metrics = [
        ("PR-AUC",         "Primary — not corrupted by true-negative majority; summarises precision-recall trade-off"),
        ("ROC-AUC",        "Secondary — overall ranking ability (but FPR denominator inflated by TN majority)"),
        ("MCC",            "Single symmetric metric using all four confusion-matrix cells"),
        ("Top-k lift",     "Event rate in top 5%/10% risk bucket vs base rate — most intuitive business metric"),
        ("FP/TP burden",   "False positives per true positive — critical for alert-noise assessment"),
        ("Brier score",    "Calibration quality — are predicted probabilities reliable?"),
    ]
    for i, (m, d) in enumerate(metrics):
        y = 2.88 + i * 0.43
        rect(sl, 0.5, y, 1.7, 0.35, TEAL if i%2==0 else NAVY)
        txt(sl, m, 0.58, y+0.06, 1.55, 0.26, size=11, bold=True, color=WHITE)
        txt(sl, d, 2.35, y+0.06, 7.3, 0.28, size=11, color=DARK)

    # ---- 10. Model Results & Recommended Model --------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Model Results & Recommended Overall Model",
              "Test-split PR-AUC · Composite score selects Logistic Regression / global")
    pg(sl, 10, TOTAL)

    # Bar chart
    cd = ChartData()
    cd.categories = ["LogReg", "ElasticNet", "Bal.RF", "XGBoost", "RF", "Rule-Based"]
    cd.add_series("PR-AUC (test)", [0.128, 0.128, 0.108, 0.109, 0.101, 0.096])
    ch = sl.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.4), Inches(1.2), Inches(5.5), Inches(3.6), cd
    ).chart
    ch.chart_title.has_text_frame = False
    ch.has_legend = False

    callout(sl, 6.15, 1.25, 3.55, 1.4,
            "Top composite model",
            "Logistic Regression / global\nPR-AUC: 0.128  |  ROC-AUC: 0.615\nTop-5% lift: 2.05×  |  MCC: 0.077")
    callout(sl, 6.15, 2.78, 3.55, 1.0,
            "Rule-based sanity check",
            "PR-AUC 0.096 — ML must beat this.\nXGBoost (0.109), Balanced RF (0.108): above baseline.",
            hc=MUTED)
    callout(sl, 6.15, 3.90, 3.55, 0.85,
            "Honest base rate",
            "Random classifier PR-AUC ≈ 0.085.\nBest gain: +50% above random.",
            hc=CORAL)

    rect(sl, 0.4, 4.93, 9.2, 0.45, RGBColor(0xE0, 0xF2, 0xFE))
    txt(sl, "Composite score formula: PR-AUC 35% + ROC-AUC 20% + MCC 20% − FP/TP burden 15% − Brier 10%  ·  Fully documented in Feature Audit tab",
        0.55, 4.97, 9.0, 0.36, size=10, color=NAVY)

    # ---- 11. Baseline vs Transformed: Three Comparability Groups ----------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Feature Transformations: Three Model Groups",
              "Delta metrics are only valid for models present in BOTH pipelines")
    pg(sl, 11, TOTAL)

    txt(sl, "Transformations change predictor variable representation, not model algorithms.",
        0.5, 1.15, 9.2, 0.3, size=13, bold=True, color=NAVY)

    # Group 1: Comparable (delta chart)
    rect(sl, 0.4, 1.52, 9.2, 0.28, TEAL)
    txt(sl, "GROUP 1 — Before/After comparison (delta is valid): Logistic Reg, Elastic Net, Random Forest, XGBoost, Rule-Based Score",
        0.55, 1.56, 8.9, 0.22, size=10, bold=True, color=WHITE)

    cd2 = ChartData()
    cd2.categories = ["XGBoost\nglobal", "RF\nglobal", "XGBoost\nbroad ETF", "LogReg\nglobal", "ElasticNet\nglobal"]
    cd2.add_series("ΔPR-AUC (transformed − baseline)", [+0.010, +0.015, +0.007, -0.002, -0.001])
    ch2 = sl.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.4), Inches(1.85), Inches(5.5), Inches(2.3), cd2
    ).chart
    ch2.chart_title.has_text_frame = False
    ch2.has_legend = False

    # Group 2 & 3 boxes on right
    rect(sl, 6.1, 1.85, 3.6, 1.1, RGBColor(0xFE, 0xF9, 0xEE), border=True)
    txt(sl, "GROUP 2 — Baseline-only", 6.22, 1.91, 3.35, 0.28, size=11, bold=True, color=CORAL)
    txt(sl, "Balanced Random Forest\nNo transformed version — delta not calculated.",
        6.22, 2.18, 3.35, 0.68, size=10, color=DARK)

    rect(sl, 6.1, 3.07, 3.6, 1.08, RGBColor(0xFE, 0xF0, 0xD8), border=True)
    txt(sl, "GROUP 3 — Transformed-only", 6.22, 3.13, 3.35, 0.28, size=11, bold=True, color=CORAL)
    txt(sl, "LightGBM\nNo baseline version — delta not calculated.\nPR-AUC: 0.119 (broad ETF), 0.113 (global)\nCompetitive with XGBoost, not a before/after result.",
        6.22, 3.40, 3.35, 0.68, size=10, color=DARK)

    rect(sl, 0.4, 4.22, 9.2, 0.55, RGBColor(0xE0, 0xF2, 0xFE))
    txt(sl, "Key finding: Tree models (XGBoost, RF) gain +0.01–0.015 PR-AUC from transformations. "
            "Linear models gain ~0 (regularisation already handles scale). "
            "No model crosses PR-AUC 0.20 in either pipeline.",
        0.55, 4.26, 8.9, 0.44, size=11, color=NAVY)

    rect(sl, 0.4, 4.85, 9.2, 0.55, RGBColor(0xFE, 0xF9, 0xEE))
    txt(sl, "Caveat: transformations improve predictor representation. They do not create new independent historical bubble events.",
        0.55, 4.90, 8.9, 0.42, size=11, italic=True, color=CORAL)

    # ---- 12. Feature Importance ----------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Feature Importance & Interpretability",
              "Cross-model convergence on the same feature families builds confidence")
    pg(sl, 12, TOTAL)
    bullets(sl, [
        "Logistic Regression signed coefficients: positive = raises risk; negative = lowers it",
        "Tree feature importance: mean impurity decrease across splits",
        "Both methods converge on: momentum, RSI extremes, rule-based scores, short-term volatility",
        "Valuation multiples (P/E, P/S) contribute — but secondary to price-action features",
        "Transformed pipeline: rolling z-scores and regime flags rank highly for XGBoost and RF",
        "Theme roll-up aggregates 100+ features into 8 interpretable families for stakeholder presentation",
    ], 0.55, 1.25, 9.1, 3.2, size=14)
    rect(sl, 0.5, 4.6, 9.0, 0.7, RGBColor(0xE0, 0xF2, 0xFE))
    txt(sl, "Cross-model agreement on feature families is evidence that the relationship is real, not model-specific. "
            "A feature that matters to Logistic Regression coefficients AND XGBoost impurity gain is likely a genuine "
            "statistical signal — not an artifact of model architecture.",
        0.65, 4.65, 8.7, 0.58, size=11, color=NAVY)

    # ---- 13. Multi-Layer Validation ------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Multi-Layer Validation",
              "Four independent evaluation angles — not just the ML chronological split")
    pg(sl, 13, TOTAL)
    layers = [
        (NAVY, "1  Chronological ML",  "70/15/15 date split. Primary. Same tickers may appear in train and test."),
        (TEAL, "2  Event-Level Valid.","Train before crisis; test on pre-event window. Fewer obs, most honest generalization."),
        (MINT, "3  Hazard Model",      "Cox-PH: conditional drawdown probability given time-in-regime. Different question."),
        (RGBColor(0x0E, 0x78, 0x9E), "4  Poisson Count", "Expected drawdown events in next N weeks. Complements point-in-time classifier."),
        (MUTED, "5  Rule-Based Sanity", "Hindenburg Omen, CAPE, RSI extremes. PR-AUC 0.096 — ML must beat this."),
    ]
    for i, (vc, title, body) in enumerate(layers):
        y = 1.22 + i * 0.8
        rect(sl, 0.5, y, 0.07, 0.6, vc)
        txt(sl, title, 0.7, y+0.02, 3.5, 0.3, size=13, bold=True, color=NAVY)
        txt(sl, body,  0.7, y+0.32, 9.0, 0.35, size=11, color=DARK)

    # ---- 14. Limitations -----------------------------------------------
    sl = prs.slides.add_slide(blank)
    light_slide(sl)
    title_bar(sl, "Limitations: Honest Assessment",
              "Weak results are not hidden — they are part of the conclusion")
    pg(sl, 14, TOTAL)
    limits = [
        (CORAL, "Small true-event count",
         "~15–30 independent bubble episodes globally. Wide PR-AUC confidence intervals. "
         "597K rows ≠ 597K independent observations."),
        (CORAL, "Implicit data snooping",
         "Iterative feature engineering on the same dataset creates overfitting risk even with "
         "chronological splits. Event-level validation partially mitigates this."),
        (CORAL, "Modest absolute performance",
         "Best PR-AUC 0.128 vs 0.085 base rate. Real signal — but not a trading edge, not a crash alarm. "
         "Top-5% bucket is 82.5% false positives."),
        (CORAL, "Survivorship & coverage bias",
         "Ticker list reflects assets that survived and were liquid. 1927–1960 data is sparse for many tickers."),
        (CORAL, "No live deployment validation",
         "ml_inference.py simulates live scoring. No paper-trading or out-of-sample deployment validation "
         "was performed."),
    ]
    for i, (vc, title, body) in enumerate(limits):
        y = 1.22 + i * 0.82
        rect(sl, 0.5, y, 0.06, 0.6, vc)
        txt(sl, title, 0.7, y+0.02, 3.5, 0.28, size=13, bold=True, color=RGBColor(0xC0, 0x30, 0x30))
        txt(sl, body,  0.7, y+0.30, 9.0, 0.40, size=11, color=DARK)

    # ---- 15. Conclusion ------------------------------------------------
    sl = prs.slides.add_slide(blank)
    dark_slide(sl)
    txt(sl, "Conclusion", 0.6, 0.28, 8.8, 0.45, size=16, color=ICE)
    txt(sl, "What the Project Shows", 0.6, 0.68, 9.0, 0.65, size=32, bold=True, color=WHITE)
    rect(sl, 0.6, 1.38, 8.8, 0.04, MINT)
    pg(sl, 15, TOTAL, dark=True)
    takeaways = [
        "A regularised linear model (Logistic Regression) is the most defensible baseline and recommended overall model — PR-AUC 0.128 vs 0.085 base rate",
        "XGBoost and Balanced RF beat the rule-based baseline but do not outperform Logistic Regression — complexity not justified on this event count",
        "Feature transformations give tree models +0.01–0.015 PR-AUC; LightGBM (Phase 8 only) is competitive with XGBoost but not superior",
        "Top-5% risk ranking enriches drawdown events ~2× — modest but real; 82.5% of flagged rows still had no drawdown",
        "Modest PR-AUC from a genuinely difficult rare-event problem is a legitimate, honest ML conclusion",
    ]
    for i, t in enumerate(takeaways):
        y = 1.55 + i * 0.60
        rect(sl, 0.55, y+0.08, 0.3, 0.3, MINT)
        txt(sl, str(i+1), 0.58, y+0.08, 0.28, 0.3, size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        txt(sl, t, 1.0, y+0.04, 8.5, 0.52, size=12, color=ICE)
    rect(sl, 0.5, 4.88, 9.0, 0.52, RGBColor(0x11, 0x2B, 0x6E))
    txt(sl, "Educational research project. The dashboard is the interpretation layer. "
            "The ML pipeline — dataset, labels, splits, multi-model evaluation, rare-event metrics — is the deliverable. "
            "Nothing here is investment advice.",
        0.65, 4.93, 8.7, 0.40, size=11, italic=True, color=ICE)

    prs.save(str(out))
    print(f"Saved: {out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    out = Path(__file__).parent / "Economic_Bubble_Dashboard_Presentation.pptx"
    build(out)
