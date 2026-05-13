"""Build Economic Bubble Dashboard presentation with python-pptx.

Run from the project root:
    python presentation/build_pptx.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import ChartData
from pptx import Presentation
from pptx.enum.chart import XL_CHART_TYPE

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY   = RGBColor(0x1E, 0x27, 0x61)   # dominant dark
TEAL   = RGBColor(0x1C, 0x72, 0x93)   # secondary
MINT   = RGBColor(0x0D, 0x94, 0x88)   # accent / positive
CORAL  = RGBColor(0xF9, 0x61, 0x67)   # warning/negative
ICE    = RGBColor(0xCA, 0xDC, 0xFC)   # light background tint
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
OFF_W  = RGBColor(0xF4, 0xF7, 0xFB)   # slide bg for content slides
DARK_T = RGBColor(0x1E, 0x29, 0x3B)   # body text

W  = Inches(10)
H  = Inches(5.625)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hex_rgb(r, g, b): return RGBColor(r, g, b)

def fill_solid(shape, color: RGBColor):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color

def add_text_box(slide, text, x, y, w, h, *,
                 font_size=18, bold=False, color=DARK_T,
                 align=PP_ALIGN.LEFT, italic=False, font_name="Calibri"):
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    return txb

def add_rect(slide, x, y, w, h, color: RGBColor, line_color=None):
    from pptx.util import Pt as PtU
    shp = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE = 1
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    fill_solid(shp, color)
    if line_color:
        shp.line.color.rgb = line_color
        shp.line.width = PtU(0.5)
    else:
        shp.line.fill.background()
    return shp

def dark_slide(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY

def light_slide(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = OFF_W

def add_slide_number(slide, n, total, dark=False):
    c = ICE if dark else RGBColor(0x94, 0xA3, 0xB8)
    add_text_box(slide, f"{n}/{total}", 9.4, 5.2, 0.5, 0.3,
                 font_size=10, color=c, align=PP_ALIGN.RIGHT)

def add_title_bar(slide, title_text, subtitle=None):
    """Teal left accent bar + title text on light slides."""
    add_rect(slide, 0.45, 0.25, 0.07, 0.7 if not subtitle else 0.9, TEAL)
    add_text_box(slide, title_text, 0.62, 0.20, 9.0, 0.5,
                 font_size=26, bold=True, color=NAVY, font_name="Calibri")
    if subtitle:
        add_text_box(slide, subtitle, 0.62, 0.68, 9.0, 0.3,
                     font_size=13, color=TEAL, font_name="Calibri")

def bullet_items(slide, items: list[str], x, y, w, h,
                 font_size=14, color=DARK_T, indent=False):
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = txb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(3)
        run = p.add_run()
        run.text = ("    " if indent else "") + "▸  " + item
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        run.font.name = "Calibri"

def stat_card(slide, x, y, w, h, value, label, value_color=MINT):
    add_rect(slide, x, y, w, h, WHITE)
    add_rect(slide, x, y, w, 0.06, value_color)  # top accent
    add_text_box(slide, value, x + 0.1, y + 0.15, w - 0.2, 0.55,
                 font_size=30, bold=True, color=value_color,
                 align=PP_ALIGN.CENTER)
    add_text_box(slide, label, x + 0.05, y + 0.70, w - 0.1, 0.4,
                 font_size=11, color=RGBColor(0x64, 0x74, 0x8B),
                 align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def build(out_path: Path):
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    blank_layout = prs.slide_layouts[6]  # completely blank
    TOTAL = 16

    # ------------------------------------------------------------------ 1. Title
    sl = prs.slides.add_slide(blank_layout)
    dark_slide(sl)
    # Large title
    add_text_box(sl, "Economic Bubble Monitoring", 0.7, 0.7, 8.6, 0.8,
                 font_size=40, bold=True, color=WHITE, font_name="Calibri",
                 align=PP_ALIGN.CENTER)
    add_text_box(sl, "& Investment Strategy Dashboard",  0.7, 1.45, 8.6, 0.7,
                 font_size=32, bold=True, color=ICE, font_name="Calibri",
                 align=PP_ALIGN.CENTER)
    # Accent line replaced by teal box
    add_rect(sl, 3.5, 2.22, 3.0, 0.04, MINT)
    add_text_box(sl, "Machine Learning for Rare Economic Events  ·  LANL / Academic Research",
                 0.5, 2.45, 9.0, 0.45,
                 font_size=14, color=ICE, align=PP_ALIGN.CENTER)
    add_text_box(sl, "Educational research project  —  not investment advice",
                 1.0, 3.05, 8.0, 0.35,
                 font_size=11, italic=True, color=RGBColor(0x8B, 0xA4, 0xD4),
                 align=PP_ALIGN.CENTER)
    add_text_box(sl, "Mark Crisci  ·  2026",
                 0.5, 4.85, 9.0, 0.4,
                 font_size=12, color=RGBColor(0x8B, 0xA4, 0xD4),
                 align=PP_ALIGN.CENTER)
    add_slide_number(sl, 1, TOTAL, dark=True)

    # ------------------------------------------------------------------ 2. Research Question
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "The Research Question",
                  "Can historical market data identify conditions that precede major drawdowns?")
    add_slide_number(sl, 2, TOTAL)
    bullet_items(sl, [
        "Economic bubbles share recognisable signatures: extreme valuation, momentum divergence, volatility compression",
        "Forecasting the exact timing of a crash is widely regarded as near-impossible",
        "A more tractable goal: estimate the probability that current conditions RESEMBLE pre-drawdown regimes",
        "Our target: burst_6m_segment = 1 if a segment-adjusted major drawdown occurs within 26 weekly observations",
        "Segment-adjusted thresholds prevent labelling a −20% SPY move the same way as a −20% Bitcoin move",
    ], 0.55, 1.25, 9.1, 3.8, font_size=15)
    add_rect(sl, 0.5, 4.9, 9.0, 0.5, RGBColor(0xE0, 0xF2, 0xFE))
    add_text_box(sl, "KEY CONSTRAINT  ·  We build a detector, not a predictor. A high score means resemblance to history, not a guaranteed crash.",
                 0.65, 4.95, 8.7, 0.35, font_size=11,
                 color=NAVY, bold=False)

    # ------------------------------------------------------------------ 3. Dataset Architecture
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Dataset Architecture", "81 tickers · 597,806 rows · 1927 – 2026")
    add_slide_number(sl, 3, TOTAL)

    # Stat cards row
    card_y = 1.2
    for cx, val, lbl, vc in [
        (0.4,  "81",       "Tickers tracked",        TEAL),
        (2.75, "597,806",  "Weekly rows",             MINT),
        (5.1,  "99 yrs",   "Historical coverage",     TEAL),
        (7.45, "4",        "Asset segments",          MINT),
    ]:
        stat_card(sl, cx, card_y, 2.15, 1.15, val, lbl, vc)

    add_text_box(sl, "Four asset segments with segment-specific drawdown thresholds:", 0.55, 2.6, 9.0, 0.35,
                 font_size=14, bold=True, color=NAVY)

    seg_rows = [
        ("global",              "All tickers pooled",          "−25% threshold"),
        ("broad_index_etf",     "SPY, QQQ, EFA, EEM, VNQ …",  "−20% threshold"),
        ("mega_cap_ai_tech",    "AAPL, MSFT, NVDA, GOOGL …",  "−35% threshold"),
        ("speculative_high_vol","BTC, ETH, ARKK, GME …",      "−50% threshold"),
    ]
    for i, (seg, desc, thr) in enumerate(seg_rows):
        y = 3.1 + i * 0.47
        add_rect(sl, 0.5, y, 2.9, 0.38, RGBColor(0xE0, 0xF2, 0xFE))
        add_text_box(sl, seg, 0.6, y + 0.04, 2.7, 0.32, font_size=12, bold=True, color=NAVY)
        add_text_box(sl, desc, 3.55, y + 0.04, 4.0, 0.32, font_size=12, color=DARK_T)
        add_rect(sl, 7.65, y, 1.8, 0.38, TEAL)
        add_text_box(sl, thr, 7.68, y + 0.04, 1.7, 0.32, font_size=12, bold=True, color=WHITE)

    # ------------------------------------------------------------------ 4. Target Label
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Target Label: burst_6m_segment",
                  "Segment-adjusted forward drawdown labels — no random shuffling, chronological splits")
    add_slide_number(sl, 4, TOTAL)

    add_text_box(sl, "Why segment-specific thresholds?", 0.55, 1.25, 9.0, 0.35,
                 font_size=16, bold=True, color=NAVY)
    bullet_items(sl, [
        "A −20% decline in SPY (broad market) is a major correction; the same move in BTC is routine volatility",
        "Using a single threshold conflates qualitatively different events and corrupts the label distribution",
        "Segment thresholds ensure each label class reflects truly-unusual drawdowns for that asset class",
    ], 0.55, 1.6, 9.0, 1.3, font_size=14)

    add_text_box(sl, "Class balance & split design", 0.55, 2.95, 9.0, 0.35,
                 font_size=16, bold=True, color=NAVY)
    # table of rates
    rows = [
        ("Scope",               "Positive rate", "Training cutoff"),
        ("global",              "8.5%",          "70% chronological"),
        ("broad_index_etf",     "11.0%",         "70% chronological"),
        ("mega_cap_ai_tech",    "6.8%",          "70% chronological"),
        ("speculative_high_vol","10.3%",         "70% chronological"),
    ]
    for i, row in enumerate(rows):
        y = 3.35 + i * 0.39
        bg = NAVY if i == 0 else (RGBColor(0xF1, 0xF5, 0xF9) if i % 2 == 1 else WHITE)
        fc = WHITE if i == 0 else DARK_T
        for j, (cell, cw) in enumerate(zip(row, [3.0, 2.2, 3.5])):
            cx = 0.5 + sum([3.0, 2.2, 3.5][:j])
            add_rect(sl, cx, y, cw, 0.36, bg)
            add_text_box(sl, cell, cx + 0.08, y + 0.06, cw - 0.1, 0.28,
                         font_size=12, bold=(i == 0), color=fc)

    # ------------------------------------------------------------------ 5. ML Pipeline
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Machine Learning Pipeline",
                  "Chronological splits · no data leakage · 5 model families per scope")
    add_slide_number(sl, 5, TOTAL)

    # Pipeline flow boxes
    steps = [
        ("1  Data\nIngestion",       "81 tickers\n1927–2026"),
        ("2  Feature\nEngineering",  "Technical · Macro\nValuation · Rule-based"),
        ("3  ML Dataset",            "burst_6m_segment\ntarget construction"),
        ("4  Chrono\nSplit",         "70% train · 15% val\n15% test"),
        ("5  Model\nTraining",       "5 families\n4 scopes each"),
        ("6  Evaluation",            "PR-AUC · ROC-AUC\nMCC · FP/TP"),
    ]
    box_w = 1.45
    for i, (title, sub) in enumerate(steps):
        bx = 0.3 + i * 1.57
        add_rect(sl, bx, 1.3, box_w, 1.05, TEAL if i % 2 == 0 else NAVY)
        add_text_box(sl, title, bx + 0.05, 1.35, box_w - 0.1, 0.55,
                     font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(sl, sub, bx + 0.05, 1.88, box_w - 0.1, 0.44,
                     font_size=9, color=ICE, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            add_text_box(sl, "→", bx + box_w, 1.68, 0.12, 0.3,
                         font_size=16, bold=True, color=TEAL, align=PP_ALIGN.CENTER)

    add_text_box(sl, "Model families trained per scope:", 0.55, 2.6, 9.0, 0.35,
                 font_size=14, bold=True, color=NAVY)
    models = [
        ("Logistic Regression",    "Linear baseline, L2 regularised"),
        ("Elastic Net Logistic",   "L1+L2 sparsity, useful for high-dim feature sets"),
        ("Random Forest",          "Ensemble tree, captures non-linearities"),
        ("Balanced Random Forest", "Class-weight adjusted RF for imbalanced labels"),
        ("XGBoost",                "Gradient boosted trees, our most complex model"),
    ]
    for i, (m, d) in enumerate(models):
        y = 3.05 + i * 0.43
        add_rect(sl, 0.5, y, 3.2, 0.35, RGBColor(0xE0, 0xF2, 0xFE))
        add_text_box(sl, m, 0.6, y + 0.04, 3.0, 0.28, font_size=12, bold=True, color=NAVY)
        add_text_box(sl, d, 3.85, y + 0.04, 5.8, 0.28, font_size=12, color=DARK_T)

    # ------------------------------------------------------------------ 6. Model Performance (chart)
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Test-Set Model Performance",
                  "PR-AUC (primary) · ROC-AUC · Global scope, chronological hold-out 2020–2026 approx")
    add_slide_number(sl, 6, TOTAL)

    # Bar chart — PR-AUC by model (global scope, test)
    cd = ChartData()
    cd.categories = ["Logistic\nRegression", "Elastic Net\nLogistic", "Balanced\nRandom Forest",
                     "XGBoost", "Random\nForest", "Rule-Based\nBaseline"]
    cd.add_series("PR-AUC (global test)",
                  [0.128, 0.128, 0.108, 0.109, 0.101, 0.096])
    cd.add_series("ROC-AUC (global test)",
                  [0.615, 0.615, 0.592, 0.585, 0.574, 0.547])

    chart = sl.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        Inches(0.4), Inches(1.25), Inches(6.0), Inches(3.9), cd
    ).chart
    chart.chart_title.has_text_frame = False
    chart.has_legend = True
    from pptx.enum.chart import XL_LEGEND_POSITION
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM

    # Annotations on right
    add_rect(sl, 6.7, 1.25, 3.0, 1.5, RGBColor(0xE0, 0xF2, 0xFE))
    add_text_box(sl, "WOW #1", 6.85, 1.30, 2.7, 0.35,
                 font_size=13, bold=True, color=TEAL)
    add_text_box(sl,
                 "Logistic Regression equals or beats XGBoost on PR-AUC. A simple linear model outperforms a boosted ensemble on this rare-event dataset.",
                 6.85, 1.62, 2.7, 0.95, font_size=11, color=DARK_T)

    add_rect(sl, 6.7, 2.9, 3.0, 1.25, RGBColor(0xFE, 0xF9, 0xEE))
    add_text_box(sl, "HONEST BASELINE", 6.85, 2.95, 2.7, 0.35,
                 font_size=13, bold=True, color=CORAL)
    add_text_box(sl,
                 "Base positive rate is 8.5% (global). A random classifier has PR-AUC ≈ 0.085. Our best model achieves 0.128.",
                 6.85, 3.28, 2.7, 0.75, font_size=11, color=DARK_T)

    # ------------------------------------------------------------------ 7. Key Finding: Top-K Lift
    sl = prs.slides.add_slide(blank_layout)
    dark_slide(sl)
    add_slide_number(sl, 7, TOTAL, dark=True)
    add_text_box(sl, "Key Finding", 0.6, 0.3, 8.8, 0.55,
                 font_size=16, color=ICE, bold=False, font_name="Calibri")
    add_text_box(sl, "WOW #2 — Top-5% Risk Ranking Lifts Event Rate 2×",
                 0.6, 0.75, 9.0, 0.75,
                 font_size=28, bold=True, color=WHITE, font_name="Calibri")
    add_rect(sl, 0.6, 1.55, 8.8, 0.04, MINT)

    stats = [
        ("2.15×",  "Top-5% lift\nBalanced RF / mega_cap_ai_tech"),
        ("2.05×",  "Top-5% lift\nLogistic Regression / global"),
        ("~8.5%",  "Base positive rate\n(global scope)"),
        ("~17.5%", "Top-5% bucket rate\n(Logistic Regression / global)"),
    ]
    for i, (val, lbl) in enumerate(stats):
        cx = 0.5 + i * 2.35
        add_rect(sl, cx, 1.75, 2.15, 1.5, RGBColor(0x11, 0x2B, 0x6E))
        add_text_box(sl, val, cx + 0.1, 1.85, 1.95, 0.7,
                     font_size=34, bold=True, color=MINT,
                     align=PP_ALIGN.CENTER)
        add_text_box(sl, lbl, cx + 0.1, 2.55, 1.95, 0.62,
                     font_size=11, color=ICE, align=PP_ALIGN.CENTER)

    add_text_box(sl, "Interpretation: The model cannot tell you exactly when a crash will happen. "
                     "But when we sort all ticker-weeks by predicted probability and take the highest-risk 5%, "
                     "that bucket contains real drawdown events at twice the baseline rate.",
                 0.6, 3.4, 9.0, 0.95, font_size=13, color=ICE)

    add_rect(sl, 0.55, 4.5, 8.9, 0.6, RGBColor(0x11, 0x2B, 0x6E))
    add_text_box(sl, "This is risk enrichment, not certainty. 82.5% of 'high-risk' rows still had NO drawdown.",
                 0.7, 4.55, 8.6, 0.45, font_size=13, italic=True, color=CORAL)

    # ------------------------------------------------------------------ 8. Why Simple Beats Complex
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "WOW #3 — Why Simple Models Win Here",
                  "Logistic Regression matches XGBoost on PR-AUC despite 10× lower complexity")
    add_slide_number(sl, 8, TOTAL)

    reasons = [
        ("Small true-event count",
         "Only ~500–2,700 positive rows in the global test set. Tree ensembles need more signal to outperform linear models."),
        ("Temporal autocorrelation",
         "Bubble regimes persist for months. Many 'events' are the same episode repeated across tickers. Independent observations are far fewer than row count suggests."),
        ("Feature linearity",
         "Many bubble precursors (momentum, RSI distance, yield spread) have near-linear relationships with risk. No need for XGBoost's non-linear capacity."),
        ("Regularisation advantage",
         "L2 / Elastic Net regularisation prevents overfitting on a relatively small unique-event count. XGBoost without careful tuning can overfit."),
    ]
    for i, (title, body) in enumerate(reasons):
        y = 1.25 + i * 0.99
        add_rect(sl, 0.5, y, 0.06, 0.8, MINT)
        add_text_box(sl, title, 0.7, y, 3.5, 0.38, font_size=14, bold=True, color=NAVY)
        add_text_box(sl, body, 0.7, y + 0.38, 8.8, 0.52, font_size=12, color=DARK_T)

    add_rect(sl, 0.5, 5.2, 9.0, 0.22, ICE)
    add_text_box(sl, "Lesson: Always train a regularised linear baseline first. Complexity is justified only when it demonstrably improves out-of-time metrics.",
                 0.6, 5.22, 8.8, 0.18, font_size=10, italic=True, color=NAVY)

    # ------------------------------------------------------------------ 9. Rare Event Challenge
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "The Rare Event Challenge",
                  "Imbalanced labels · SMOTE experiments · Balanced Random Forest · Cost thresholds")
    add_slide_number(sl, 9, TOTAL)

    left = [
        "8.5% positive rate (global) — standard classifiers optimise for the majority class",
        "Accuracy is misleading: predicting 'never burst' gives 91.5% accuracy with 0% recall",
        "PR-AUC preferred: summarises precision-recall trade-off, not corrupted by TN-majority",
        "MCC: single metric that accounts for all four confusion-matrix cells symmetrically",
    ]
    right = [
        "SMOTE / SMOTE-ENN: oversample minority class in training — improves recall but risks FP inflation",
        "Balanced Random Forest: class_weight='balanced' equivalent for trees — robust alternative",
        "Cost-ratio thresholds: vary FP/FN cost ratio to find operating points for different use cases",
        "Rare-event constrained threshold: minimum precision ≥ 12%, alert rate ≤ 20% enforced",
    ]
    add_text_box(sl, "Problem", 0.55, 1.2, 4.5, 0.3, font_size=14, bold=True, color=NAVY)
    add_text_box(sl, "Our Approach", 5.3, 1.2, 4.5, 0.3, font_size=14, bold=True, color=NAVY)
    bullet_items(sl, left,  0.5, 1.55, 4.5, 3.5, font_size=12)
    bullet_items(sl, right, 5.2, 1.55, 4.5, 3.5, font_size=12)
    add_rect(sl, 4.9, 1.15, 0.03, 4.1, ICE)

    # ------------------------------------------------------------------ 10. Phase 8 Transformations
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Phase 8: Feature Transformation Experiment",
                  "Controlled experiment: same splits, same models, only the feature pipeline changes")
    add_slide_number(sl, 10, TOTAL)

    add_text_box(sl, "9 transformation families · 88 new columns on top of baseline",
                 0.55, 1.2, 9.0, 0.35, font_size=14, bold=True, color=TEAL)

    families = [
        ("log1p / signed-log",         "Compress right-skewed valuation metrics"),
        ("Winsorisation",               "Remove extreme outliers (fit on TRAIN only)"),
        ("Rolling z-score by ticker",   "Normalise within-ticker time series"),
        ("Expanding percentile rank",   "Non-parametric rank transform, shift(1) leakage guard"),
        ("Regime flags",                "Binary indicators from volatility/RSI thresholds"),
        ("Interaction features",        "PE × momentum, vol × drawdown cross terms"),
        ("Vol-adjusted returns",        "Return scaled by rolling 13w realised vol"),
        ("Drawdown-state flags",        "Binary: currently in drawdown ≥ X%?"),
        ("Squared terms",               "Captures non-linear tails in linear models"),
    ]
    col1 = families[:5]
    col2 = families[5:]
    for i, (name, desc) in enumerate(col1):
        y = 1.7 + i * 0.60
        add_rect(sl, 0.5, y, 0.06, 0.44, MINT)
        add_text_box(sl, name, 0.68, y, 2.5, 0.25, font_size=11, bold=True, color=NAVY)
        add_text_box(sl, desc, 0.68, y + 0.24, 4.1, 0.22, font_size=10, color=DARK_T)
    for i, (name, desc) in enumerate(col2):
        y = 1.7 + i * 0.60
        add_rect(sl, 5.2, y, 0.06, 0.44, TEAL)
        add_text_box(sl, name, 5.38, y, 2.5, 0.25, font_size=11, bold=True, color=NAVY)
        add_text_box(sl, desc, 5.38, y + 0.24, 4.3, 0.22, font_size=10, color=DARK_T)

    add_rect(sl, 0.5, 4.75, 9.0, 0.55, RGBColor(0xE0, 0xF2, 0xFE))
    add_text_box(sl,
                 "Leakage guard: Winsor bounds and regime thresholds fitted on TRAIN split only (TransformFitState). "
                 "Rolling z-scores use shift(1) before expanding window to prevent look-ahead.",
                 0.65, 4.82, 8.7, 0.40, font_size=11, color=NAVY)

    # ------------------------------------------------------------------ 11. Transformation Results
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "WOW #4 — Transformation Results",
                  "Selective gains: linear models benefit, trees less so; consistent pattern matters more than magnitude")
    add_slide_number(sl, 11, TOTAL)

    # Chart: delta PR-AUC
    cd2 = ChartData()
    cd2.categories = ["XGBoost\nglobal", "Random Forest\nglobal", "Balanced RF\nglobal",
                      "Logistic Reg\nglobal", "Elastic Net\nglobal",
                      "XGBoost\nbroad ETF", "Random Forest\nbroad ETF"]
    cd2.add_series("ΔPR-AUC (transformed − baseline)",
                   [+0.010, +0.015, 0.000, -0.002, -0.001,
                    +0.007, +0.015])
    chart2 = sl.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        Inches(0.4), Inches(1.25), Inches(5.8), Inches(3.7), cd2
    ).chart
    chart2.chart_title.has_text_frame = False
    chart2.has_legend = False

    # Right panel takeaways
    add_rect(sl, 6.45, 1.25, 3.3, 3.7, RGBColor(0xF1, 0xF5, 0xF9))
    add_text_box(sl, "Takeaways", 6.6, 1.35, 3.0, 0.35,
                 font_size=14, bold=True, color=NAVY)
    tks = [
        "Tree models (XGBoost, RF) gain +0.01–0.015 PR-AUC from transformations",
        "Linear models show near-zero delta — they already handle scale via regularisation",
        "LightGBM (Phase 8 benchmark) PR-AUC 0.113 on broad ETF — competitive with XGBoost",
        "No model crosses the 'practically useful' bar of PR-AUC > 0.20",
        "A consistent positive delta (not cherry-picked) is the honest interpretation",
    ]
    bullet_items(sl, tks, 6.5, 1.7, 3.2, 3.0, font_size=11)

    add_rect(sl, 0.4, 5.1, 9.2, 0.35, ICE)
    add_text_box(sl, "Honest framing: transformations improve feature representation — they do not create new independent historical bubble events.",
                 0.55, 5.13, 9.0, 0.28, font_size=10, italic=True, color=NAVY)

    # ------------------------------------------------------------------ 12. Multi-Layer Validation
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Multi-Layer Validation Framework",
                  "Chronological ML · Event-level leave-one-crisis · Hazard model · Poisson count model · Rule-based baseline")
    add_slide_number(sl, 12, TOTAL)

    layers = [
        ("Chronological ML",  "70/85 date split, no shuffle. Primary evaluation method. Temporally sound but same tickers across splits."),
        ("Event-Level Valid.", "Train before a historical crisis, test on the pre-event window. Fewer observations but the honest generalization test."),
        ("Hazard Model",      "Cox-PH model: estimates conditional probability of drawdown given time-in-regime. Complements point-in-time ML scores."),
        ("Poisson Count",     "Count model: how many drawdown events expected in the next N weeks? Different question, complementary answer."),
        ("Rule-Based Baseline", "Hindenburg Omen, RSI extremes, CAPE excess, yield inversion. PR-AUC ≈ 0.096 — ML must beat this to justify complexity."),
    ]
    for i, (title, body) in enumerate(layers):
        y = 1.25 + i * 0.82
        color = [NAVY, TEAL, MINT, RGBColor(0x0E, 0x78, 0x9E), RGBColor(0x64, 0x74, 0x8B)][i]
        add_rect(sl, 0.5, y, 0.07, 0.62, color)
        add_text_box(sl, title, 0.7, y, 3.0, 0.3, font_size=13, bold=True, color=NAVY)
        add_text_box(sl, body, 0.7, y + 0.30, 9.0, 0.38, font_size=11, color=DARK_T)

    # ------------------------------------------------------------------ 13. Dashboard Overview
    sl = prs.slides.add_slide(blank_layout)
    dark_slide(sl)
    add_slide_number(sl, 13, TOTAL, dark=True)
    add_text_box(sl, "14-Section Interactive Dashboard", 0.5, 0.25, 9.5, 0.55,
                 font_size=28, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(sl, "Streamlit · python dashboard.py  ·  http://localhost:8502",
                 0.5, 0.80, 9.5, 0.38, font_size=14, color=ICE, align=PP_ALIGN.CENTER)

    pages_left = [
        "1.  Executive Summary & Story",
        "2.  Data Overview & EDA",
        "3.  Bubble Explorer",
        "4.  Historical Comparison",
        "5.  Bubble Vital Signs",
        "6.  Current AI Cycle Monitor",
        "7.  ML Model Lab  ★",
    ]
    pages_right = [
        "8.   Rare Event & Imbalance Lab",
        "9.   Event-Level Validation",
        "10. Hazard & Poisson Models",
        "11. Investment Simulator",
        "12. Methods & Diagrams",
        "13. Literature Research",
        "14. Data Quality & Limitations",
    ]
    add_rect(sl, 0.4, 1.35, 4.55, 3.8, RGBColor(0x11, 0x2B, 0x6E))
    add_rect(sl, 5.1, 1.35, 4.55, 3.8, RGBColor(0x11, 0x2B, 0x6E))
    for i, t in enumerate(pages_left):
        y = 1.5 + i * 0.48
        add_text_box(sl, t, 0.6, y, 4.1, 0.40, font_size=12, color=ICE if "★" not in t else WHITE,
                     bold=("★" in t))
    for i, t in enumerate(pages_right):
        y = 1.5 + i * 0.48
        add_text_box(sl, t, 5.25, y, 4.1, 0.40, font_size=12, color=ICE)

    add_text_box(sl, "★  ML Lab includes: baseline toggle · Transformed vs Baseline · Feature Audit · Recommended Model",
                 0.5, 5.2, 9.5, 0.3, font_size=10, italic=True, color=RGBColor(0x8B, 0xA4, 0xD4),
                 align=PP_ALIGN.CENTER)

    # ------------------------------------------------------------------ 14. Limitations
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Limitations & Honest Assessment",
                  "What we can claim · What we cannot · What was out of scope")
    add_slide_number(sl, 14, TOTAL)

    limits = [
        ("Small true-event count",
         "Despite 597K rows, the number of truly independent bubble events is perhaps 15–30. "
         "Statistical power is limited. Confidence intervals on PR-AUC are wide."),
        ("Survivorship & selection bias",
         "Ticker list reflects assets that existed and were liquid enough to track. "
         "Many assets from the 1927–1960 period are missing. Results may not generalise."),
        ("Data snooping risk",
         "Iterative feature engineering on the same dataset can implicitly overfit even with "
         "chronological splits. Event-level validation helps but does not fully eliminate this."),
        ("Modest absolute performance",
         "Best test PR-AUC is 0.128 vs 0.085 base rate. This is real but modest signal — not a "
         "trading edge, not a crash alarm, not actionable without further validation."),
        ("Live deployment not attempted",
         "The dashboard simulates live scoring (ml_inference.py). No paper-trading validation "
         "was performed. Real-world deployment would require data pipeline hardening."),
    ]
    for i, (title, body) in enumerate(limits):
        y = 1.25 + i * 0.82
        add_rect(sl, 0.5, y, 0.06, 0.60, CORAL)
        add_text_box(sl, title, 0.7, y, 3.5, 0.28, font_size=13, bold=True, color=RGBColor(0xC0, 0x30, 0x30))
        add_text_box(sl, body, 0.7, y + 0.28, 9.0, 0.40, font_size=11, color=DARK_T)

    # ------------------------------------------------------------------ 15. Contributions
    sl = prs.slides.add_slide(blank_layout)
    light_slide(sl)
    add_title_bar(sl, "Technical Contributions",
                  "Pipeline · Tooling · Evaluation framework · Educational design")
    add_slide_number(sl, 15, TOTAL)

    contribs = [
        ("End-to-end pipeline",          "14-script reproducible pipeline from raw API calls to scored dashboard"),
        ("Segment-adjusted labels",       "Segment-specific drawdown thresholds — a principled improvement over single-threshold labelling"),
        ("Phase 8 leakage-proof transforms", "TransformFitState: winsor/regime fits on TRAIN only, shift(1) lookback guard in all rolling features"),
        ("Multi-layer evaluation",        "Chronological + event-level + hazard + Poisson — four independent angles on the same question"),
        ("Composite model scorer",        "Pre-specified weighted scoring formula across 5 metrics; auditable, replicable, not cherry-picked"),
        ("14-tab narrative dashboard",    "Each page includes interpretation boxes, regime badges, missing-artifact notices, expanders with methodology"),
        ("Honest framing throughout",     "Every chart includes a 'do not overclaim' interpretation box; base rates shown alongside all lift metrics"),
    ]
    for i, (title, body) in enumerate(contribs):
        y = 1.25 + i * 0.60
        add_rect(sl, 0.5, y, 0.06, 0.44, MINT if i % 2 == 0 else TEAL)
        add_text_box(sl, title, 0.7, y, 3.6, 0.25, font_size=12, bold=True, color=NAVY)
        add_text_box(sl, body, 0.7, y + 0.24, 9.0, 0.28, font_size=11, color=DARK_T)

    # ------------------------------------------------------------------ 16. Conclusion
    sl = prs.slides.add_slide(blank_layout)
    dark_slide(sl)
    add_slide_number(sl, 16, TOTAL, dark=True)
    add_text_box(sl, "Conclusion", 0.6, 0.3, 8.8, 0.5,
                 font_size=16, color=ICE)
    add_text_box(sl, "What We Learned", 0.6, 0.75, 9.0, 0.65,
                 font_size=32, bold=True, color=WHITE)
    add_rect(sl, 0.6, 1.42, 8.8, 0.04, MINT)

    takeaways = [
        "Logistic Regression with chronological splits is the most defensible baseline — simpler models survive honest evaluation better than complex ones",
        "Top-5% risk ranking enriches the event rate ~2×, which is modest but real signal on 97 years of data",
        "Feature transformations provide +0.01–0.015 PR-AUC for tree models — worthwhile but not game-changing",
        "The honest answer is that predicting bubble bursts is hard and our modest PR-AUC reflects that honestly",
        "The dashboard infrastructure — interpretation boxes, regime badges, multi-layer validation, missing-artifact notices — is the project's most durable contribution",
    ]
    for i, t in enumerate(takeaways):
        y = 1.6 + i * 0.60
        add_rect(sl, 0.55, y + 0.08, 0.3, 0.3, MINT)
        add_text_box(sl, str(i + 1), 0.58, y + 0.08, 0.28, 0.3,
                     font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_text_box(sl, t, 1.0, y, 8.5, 0.52, font_size=13, color=ICE)

    add_rect(sl, 0.5, 4.88, 9.0, 0.55, RGBColor(0x11, 0x2B, 0x6E))
    add_text_box(sl,
                 "This is an educational research project. Nothing here constitutes investment advice. "
                 "All findings should be validated on fresh out-of-sample data before any application.",
                 0.65, 4.93, 8.7, 0.40, font_size=11, italic=True, color=ICE)

    # ------------------------------------------------------------------
    prs.save(str(out_path))
    print(f"Saved: {out_path}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    out = Path(__file__).parent / "Economic_Bubble_Dashboard_Presentation.pptx"
    build(out)
