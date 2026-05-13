"""Long-form explanations and glossary for the dashboard.

Everything in this file is text. Each function either returns a markdown string
or renders an explanation directly into Streamlit (as an `st.expander` so the
explanation never crowds the data view).

Two audiences are addressed throughout:

- **Non-experts** who want a plain-English description of what a bubble is,
  what a model is doing, and what the chart on screen is telling them.
- **ML professors / reviewers** who need rigorous metric definitions, the
  reason a metric was chosen, and the standard "but" — what it can be fooled
  by.

The text is intentionally cautious. Anywhere we mention a model "score" we
also say it is not a literal probability. Anywhere we describe a bubble we
also note that bubbles are only labelled with hindsight.
"""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Project framing
# ---------------------------------------------------------------------------
PROJECT_DISCLAIMER = (
    "**Educational research tool only.** This dashboard studies historical "
    "bubble-like market regimes and estimates whether current or past asset "
    "conditions resemble historically higher-risk drawdown regimes. It is a "
    "research signal and model-interpretation tool, not a trading oracle. "
    "Nothing here is personalized financial advice."
)


def project_purpose_md() -> str:
    return (
        "### Research question\n\n"
        "**Can historical bubble-like market patterns help identify data-driven "
        "warning, risk-ranking, and re-entry signals for a possible current or "
        "future AI-related market cycle?**\n\n"
        "The dashboard studies past regimes (dot-com, housing/credit, crypto cycles, "
        "COVID-era speculation, the AI / mega-cap cycle) and asks whether transparent "
        "historical rules — plus supervised ML on weekly features — can flag "
        "*resemblance* to higher-risk drawdown environments. The framing is "
        "**ranking and resemblance, not prophecy.**"
    )


def what_we_can_and_cannot_answer_md() -> str:
    return (
        "#### What the dashboard *can* answer\n"
        "- How did different assets behave before, during, and after historical bubble peaks?\n"
        "- Which technical and macro warning signals appeared before or during breakdowns?\n"
        "- Which historical risk regimes does today's AI / mega-cap cycle most resemble?\n"
        "- Which re-entry rules backtested better or worse after large historical drawdowns?\n\n"
        "#### What it explicitly *cannot* answer\n"
        "- The exact market top or bottom.\n"
        "- Whether the current AI cycle is or is not a bubble.\n"
        "- Whether any specific user should buy, sell, or hold an asset.\n"
        "- Whether a historical signal will continue to work in the future.\n"
    )


def what_is_a_bubble_md() -> str:
    return (
        "**What is a bubble?** Informally: prices rise far faster than any reasonable "
        "improvement in cash flows or growth could justify, followed by a sharp reversal. "
        "Formally there is no agreed definition — bubbles are usually *labelled with "
        "hindsight*. That is one of the central difficulties of this project: we only "
        "know a bubble existed after it has burst.\n\n"
        "This dashboard does **not** try to define bubbles in real time. Instead it asks: "
        "**did this week resemble historical weeks that were later followed by a major "
        "drawdown?** That is a smaller, more testable question."
    )


def why_prediction_is_hard_md() -> str:
    return (
        "**Why predicting bubble bursts is hard**\n\n"
        "1. **Few independent events.** Even with decades of weekly data per ticker, the "
        "number of *independent* historical crashes is small — perhaps a handful per "
        "asset class. Models that look impressive on a million rows may actually have "
        "seen only ~5 real crisis regimes.\n"
        "2. **Regime change.** Each crisis has different mechanics: dot-com was a "
        "valuation bubble in technology; 2008 was a credit/housing bubble; 2022 was a "
        "rates-driven repricing. Patterns from one rarely transfer cleanly to the next.\n"
        "3. **Non-stationarity.** The very features that worked best in past crises "
        "(curve inversion, leverage, valuation extremes) can stop working when central "
        "banks, regulations, or market structure change.\n"
        "4. **Path dependence.** Bubbles can stay 'overvalued' for years before "
        "reversing. A signal that fires too early is statistically a false positive "
        "even if it is economically correct.\n"
        "5. **Imbalance.** Most weeks are calm. A model that predicts 'no burst' "
        "always achieves high accuracy and is useless for our actual goal."
    )


# ---------------------------------------------------------------------------
# Asset segments
# ---------------------------------------------------------------------------
def global_vs_segment_md() -> str:
    return (
        "### Global model vs segment-specific models\n\n"
        "**Analogy.** A *global* model is like one weather forecast for the entire country. "
        "Segment models are like separate forecasts for hurricanes, snowstorms, deserts, and "
        "coastal fog. The global model sees more weather; the segment models understand local climate better.\n\n"
        "| Segment | What it covers | How it behaves | Interpretation caveat |\n"
        "|---|---|---|---|\n"
        "| **global** | All assets and segments pooled | More training data; broader risk ranking | Heterogeneous — a 'high score' means different things across segments |\n"
        "| **broad_index_etf** | SPY, QQQ, sector ETFs, indices | Drawdowns are usually smaller, macro-sensitive | Hard to move the needle — major index drawdowns are rare and macro-driven |\n"
        "| **mega_cap_ai_tech** | Profitable AI/tech leaders (NVDA, MSFT, etc.) | Volatile, but with real earnings | High valuation may be earnings-supported, so 'stretched' here ≠ 'stretched' for speculative names |\n"
        "| **speculative_high_vol** | Crypto, ARKK-style, unprofitable growth, meme-like | Much larger drawdowns common; thresholds must be harsher | Top-bucket lift looks impressive but is also noisier |\n"
        "| **other_single_name** | Individual equities that don't fit cleanly elsewhere | Mixed; least homogeneous | Use as fallback; segment-specific models are usually preferred |\n\n"
        "**Why this matters for `burst_6m_segment`.** A 20% decline in SPY is a severe event; "
        "a 20% decline in BTC is a Tuesday. Segment-adjusted thresholds let the supervised "
        "label mean the same *severity* across segments even though the *magnitudes* differ."
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def model_families_md() -> str:
    return (
        "### Model families included in the project\n\n"
        "| Model | Type | Why it's here |\n"
        "|---|---|---|\n"
        "| **Rule-Based Warning Score** | Hand-crafted scorecard | Transparent baseline; you can read off the rule that fired |\n"
        "| **Logistic Regression** | Linear classifier | Stable, interpretable; usually the best broad risk-ranker by ROC/PR-AUC |\n"
        "| **Elastic Net Logistic** | Regularized linear | Better feature selection; strong recall on AI/mega-cap segment |\n"
        "| **Random Forest** | Tree ensemble | Captures non-linear interactions |\n"
        "| **Balanced Random Forest** | Tree ensemble + class-balanced bootstraps | Built for class imbalance; useful for top-risk bucket detection |\n"
        "| **XGBoost** | Gradient-boosted trees | Standard benchmark; included to show it does *not* automatically dominate |\n"
        "| **Firth Logistic (optional, R)** | Penalized logistic | Handles small-sample / separation issues better than ordinary logistic |\n"
        "| **Discrete-time Hazard** | Event-onset model | Asks 'did the asset enter a major drawdown state in the next 4 weeks?' |\n"
        "| **Poisson Count** | Count regression | Estimates *how many* assets in a segment may enter a drawdown — segment-level, not ticker-level |\n\n"
        "**Why no neural networks (yet).** The bottleneck on this project is not model "
        "complexity. It is data scarcity (few independent crashes), non-stationarity, and "
        "label noise. A deep model would overfit unless validation is much stricter than "
        "row-level splits. The project keeps the door open but does not pretend a "
        "neural net would 'solve' a problem that is really about data structure."
    )


def why_simpler_can_beat_complex_md() -> str:
    return (
        "**Why simpler models often beat complex ones here.** Financial data is noisy, "
        "regime-dependent, and severely imbalanced. Powerful models like XGBoost have "
        "the capacity to fit the noise, which looks like better training/validation "
        "metrics but generalizes badly across crisis regimes. Regularized linear models "
        "(Logistic + Elastic Net) and balanced ensembles often produce more stable "
        "rankings on this kind of data. The rule-based baseline is included for the "
        "same reason: a model that cannot beat a transparent scorecard is not worth "
        "the complexity premium."
    )


# ---------------------------------------------------------------------------
# Metrics glossary
# ---------------------------------------------------------------------------
METRIC_GLOSSARY: dict[str, dict[str, str]] = {
    "ROC-AUC": {
        "what": "Area under the ROC curve — overall ranking quality across all thresholds.",
        "good": "Higher is better; 0.5 = chance, 1.0 = perfect.",
        "watch": (
            "ROC-AUC can look strong even when positive events are rare, because it averages over "
            "many thresholds the user would never actually use. Always pair with PR-AUC for rare events."
        ),
    },
    "PR-AUC": {
        "what": "Area under the precision-recall curve — focuses on the positive (rare) class.",
        "good": "Higher is better. Even a 'small' PR-AUC can be informative if the base rate is tiny.",
        "watch": "Cannot be compared across datasets with different base rates.",
    },
    "Accuracy": {
        "what": "Share of rows where the model's binary prediction matched the truth.",
        "good": "Higher is better — *in isolation*.",
        "watch": (
            "Accuracy is misleading when events are rare. A model that always predicts 'no burst' "
            "achieves 95%+ accuracy here and is useless for our goal."
        ),
    },
    "Precision": {
        "what": "Of the rows the model flagged as risky, what share actually had a major drawdown?",
        "good": "Higher is better. High precision = fewer false alarms.",
        "watch": "High precision usually means the model is conservative and may miss many real events.",
    },
    "Recall": {
        "what": "Of the rows that actually had a major drawdown, what share did the model catch?",
        "good": "Higher is better. High recall = fewer missed crashes.",
        "watch": "A model can game recall by flagging almost everything as risky — check alert rate and FP burden.",
    },
    "F1": {
        "what": "Harmonic mean of precision and recall.",
        "good": "Higher is better. Balances missed events vs false alarms.",
        "watch": "F1 still rewards models with very high recall on noisy alerts; combine with MCC.",
    },
    "MCC": {
        "what": "Matthews correlation coefficient — uses all four confusion-matrix cells.",
        "good": "Higher is better; 0 = chance, 1 = perfect, −1 = inverted.",
        "watch": "Harder to game with class imbalance than F1 or accuracy. Preferred for rare events.",
    },
    "Brier score": {
        "what": "Mean squared error between predicted probability and observed 0/1 outcome.",
        "good": "Lower is better. Sensitive to calibration as well as ranking.",
        "watch": "Brier is sensitive to base rate; compare within scope, not across segments.",
    },
    "False positives": {
        "what": "Rows the model flagged as risky where no major drawdown actually occurred.",
        "good": "Lower is better — fewer noisy alerts.",
        "watch": "Some FPs are 'right too early.' We cannot fully distinguish wrong from early.",
    },
    "False negatives": {
        "what": "Rows where a major drawdown happened but the model did not flag them.",
        "good": "Lower is better — fewer missed events.",
        "watch": "Rare-event models will *always* have some FN. The question is whether they cluster in a particular regime.",
    },
    "False positives per true positive": {
        "what": "Approximate cost of a single correct alert — how many false alarms accompany it.",
        "good": "Lower is better.",
        "watch": "Treat as the 'honesty meter' for alert systems. High recall with FP/TP > 20 is rarely useful.",
    },
    "Alert rate": {
        "what": "Share of rows the model flags as risky.",
        "good": "Should be roughly proportional to base rate × top-bucket size.",
        "watch": "A model flagging 50% of rows as risky is not informative regardless of recall.",
    },
    "Top-5% lift": {
        "what": "Empirical event rate in the highest-risk 5% of model scores ÷ base event rate.",
        "good": "Higher is better; 1.0 = no signal, 3.0+ = strong risk enrichment.",
        "watch": "Lift is meaningful only if the top bucket contains enough rows to be statistically stable.",
    },
    "Top-10% lift": {
        "what": "Empirical event rate in the highest-risk 10% of model scores ÷ base event rate.",
        "good": "Higher is better. Often the most operationally useful metric for risk ranking.",
        "watch": "Same caveat as top-5% — needs enough rows in the bucket.",
    },
    "Empirical bucket event rate": {
        "what": "Observed share of rows in a risk bucket that had a major forward drawdown.",
        "good": "Should rise monotonically from low to high buckets.",
        "watch": (
            "This is what to quote instead of the raw model probability when calibration is imperfect. "
            "It is the historical answer, not a forward forecast."
        ),
    },
}


def metric_explainer(metric_name: str) -> str:
    """Return a 3-line markdown explainer for a named metric."""
    info = METRIC_GLOSSARY.get(metric_name)
    if info is None:
        return f"_No glossary entry for {metric_name}._"
    return (
        f"**{metric_name}** — {info['what']}\n\n"
        f"*Direction:* {info['good']}\n\n"
        f"*Watch out for:* {info['watch']}"
    )


def full_metric_glossary_md() -> str:
    parts = ["### Metric glossary\n"]
    for name in METRIC_GLOSSARY:
        parts.append(metric_explainer(name))
        parts.append("\n---\n")
    return "\n".join(parts)


def why_accuracy_is_not_enough_md() -> str:
    return (
        "**Why accuracy alone is not enough.** Our positive class (major segment-adjusted "
        "drawdown within 26 weeks) is rare — often well under 10% of usable rows. A model "
        "that always predicts 'no burst' will look 90%+ accurate and provide zero risk "
        "ranking. That is why this dashboard puts PR-AUC, MCC, top-bucket lift, and false-"
        "positive burden ahead of accuracy in every comparison."
    )


def why_pr_auc_matters_md() -> str:
    return (
        "**Why PR-AUC matters for rare events.** ROC-AUC averages performance across "
        "all thresholds — including ones with so many false positives the model is "
        "useless in practice. PR-AUC focuses on the precision–recall tradeoff for the "
        "rare positive class, which is the regime we actually care about. A model with "
        "ROC-AUC 0.75 and PR-AUC 0.05 is *not* a useful crash-risk model."
    )


def why_calibration_matters_md() -> str:
    return (
        "**Why calibration matters.** A model score of 0.30 is meaningful only if rows with "
        "score ≈ 0.30 historically had drawdowns about 30% of the time. When the empirical "
        "rate diverges sharply from the score, the model is mis-calibrated. In this project "
        "calibration is *imperfect* — that is why the dashboard reports both the raw model "
        "score and the **empirical bucket event rate**, and prefers the empirical one for "
        "interpretation."
    )


def why_probabilities_should_not_be_overinterpreted_md() -> str:
    return (
        "**Why we don't treat scores as literal probabilities.** A 'probability' of 0.40 "
        "from a tree ensemble or an unregularized logistic is partly a ranking artifact, not "
        "a calibrated forward probability. We treat scores as *resemblance scores*: higher "
        "score = more like historical pre-drawdown weeks. Whenever you see a percentage in "
        "the AI Monitor or ML Lab, prefer the **empirical bucket rate** label."
    )


# ---------------------------------------------------------------------------
# Rare-event modeling
# ---------------------------------------------------------------------------
def why_rare_events_are_hard_md() -> str:
    return (
        "**Why rare-event modeling is fundamentally hard**\n\n"
        "- **Sample size of *events*, not rows.** The dataset may have 200,000 weekly rows, "
        "but only a few dozen distinct historical crisis episodes. Most positive rows are "
        "neighbors of the same event and are *not independent*.\n"
        "- **Class imbalance.** With base rates of 2–10%, naive cross-entropy is dominated "
        "by the negative class. Models lazily learn 'predict no burst.'\n"
        "- **Optimistic validation.** Random row splits leak information across event "
        "boundaries — the train set often contains rows just before or after the same "
        "event present in the test set. **Event-level / chronological validation** is "
        "stricter and is what this project uses."
    )


def why_big_row_count_is_misleading_md() -> str:
    return (
        "**Caveat: a huge row count does not mean many independent crashes.** "
        "If a single 2008-style episode contributes thousands of positive rows across "
        "dozens of tickers, the model effectively saw *one* crisis, not thousands of "
        "independent ones. This is why event-level validation, regime-aware splits, and "
        "leakage audits matter more than raw row totals."
    )


def smote_caveats_md() -> str:
    return (
        "**SMOTE and SMOTE-ENN: useful but easy to misread.**\n\n"
        "SMOTE synthesises new positive rows by interpolating between existing ones. "
        "It can help linear models that struggle with class imbalance, but it also:\n\n"
        "- Invents 'events' that never happened in real markets.\n"
        "- Inflates apparent recall while degrading calibration and precision.\n"
        "- Can make event-level validation look worse than ordinary validation because "
        "the synthesised positives cluster around the original crisis events.\n\n"
        "This project treats SMOTE as a **controlled experiment**, not a default. A SMOTE "
        "variant is kept only if it improves PR-AUC, MCC, top-k lift, *and* false-positive "
        "burden."
    )


def balanced_rf_vs_rf_md() -> str:
    return (
        "**Balanced Random Forest vs ordinary Random Forest.** Both fit many decision "
        "trees and average their votes. The difference is the bootstrap sample. Ordinary "
        "RF samples each tree from the full training data — so rare positives are scarce "
        "in every tree. Balanced RF undersamples the majority class for each tree, so "
        "every tree sees a roughly balanced view. This can sharply improve top-bucket "
        "lift and recall on rare-event problems *without* the synthesis artifacts of SMOTE."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def why_event_validation_matters_md() -> str:
    return (
        "**Why event-level validation matters.** Row-level chronological splits already "
        "respect time, but they still let the model learn within-event correlations. "
        "Event-level validation (sometimes called 'leave-one-crisis-out') trains on data "
        "*before* a crisis window and tests on the pre-event and event window itself. "
        "It usually produces *less flattering* metrics — which is exactly why it is the "
        "honest test of whether the model generalizes across regimes rather than across rows."
    )


def why_rule_baselines_matter_md() -> str:
    return (
        "**Why rule-based baselines matter.** If a hand-crafted scorecard built from a "
        "few transparent indicators (warning score, distance from 200DMA, weekly RSI, "
        "drawdown depth) can match or beat a tuned XGBoost, the complex model is not "
        "earning its keep. The rule-based baseline also gives non-experts a way to read "
        "*why* a signal fired — something black-box models cannot offer."
    )


# ---------------------------------------------------------------------------
# Signal confirmation
# ---------------------------------------------------------------------------
def signal_confirmation_md() -> str:
    return (
        "### Signal confirmation: agreement across independent indicators\n\n"
        "No single model on this dashboard is strong enough to be treated as a crash "
        "predictor. The research value comes from **confirmation across independent "
        "signals**:\n\n"
        "1. **Technical / rule-based warning score** (transparent, no ML).\n"
        "2. **ML risk bucket** (which historical scores the asset resembles).\n"
        "3. **Hazard / onset score** (short-horizon event-onset model).\n"
        "4. **Macro / valuation stress** (rate, curve, valuation features).\n"
        "5. **Historical top-bucket lift** (does the model's top bucket actually contain elevated event rates?).\n\n"
        "When several agree, the joint signal is stronger than any one alone. When they "
        "disagree, the dashboard intentionally reports both — the goal is honest "
        "ambiguity, not false certainty."
    )


def best_model_depends_on_goal_md() -> str:
    return (
        "### 'Best model depends on the decision goal.'\n\n"
        "Different goals select different winners:\n\n"
        "- **Best broad risk-ranking** — highest global PR-AUC / ROC-AUC. Usually "
        "**Logistic Regression / global**.\n"
        "- **Best catch-more-events model (recall)** — usually "
        "**Elastic Net Logistic / mega_cap_ai_tech** in the current run.\n"
        "- **Best cleaner warning (precision / low FP burden)** — usually the "
        "**Rule-Based Warning Score / broad_index_etf**.\n"
        "- **Best risk-bucket lift** — usually a **Balanced Random Forest** in the "
        "speculative or AI/tech segment.\n"
        "- **Best interpretable** — Rule-Based Warning Score or Logistic coefficients.\n\n"
        "There is no single 'best model.' The dashboard shows all of them because the "
        "right choice depends on what you are trying to do."
    )


# ---------------------------------------------------------------------------
# Streamlit helpers — render the explanations as expanders so they never
# overwhelm the data view.
# ---------------------------------------------------------------------------
def render_expander(label: str, body_md: str, expanded: bool = False) -> None:
    """Render a single explanation expander. Used liberally across pages."""
    with st.expander(label, expanded=expanded):
        st.markdown(body_md)


def render_project_disclaimer() -> None:
    st.warning(PROJECT_DISCLAIMER)


def render_global_vs_segment_expander() -> None:
    render_expander(
        "🧭 What do *global* and *segment* models mean? (weather analogy)",
        global_vs_segment_md(),
    )


def render_metric_glossary_expander() -> None:
    render_expander(
        "📐 Metric glossary — what every column on this page means",
        full_metric_glossary_md(),
    )


def render_rare_event_caveats_expander() -> None:
    render_expander(
        "⚠️ Why rare events are hard (and why a big row count can mislead)",
        why_rare_events_are_hard_md() + "\n\n" + why_big_row_count_is_misleading_md(),
    )


def render_signal_confirmation_expander() -> None:
    render_expander(
        "🔁 How the dashboard combines independent signals",
        signal_confirmation_md(),
    )


def render_best_model_by_goal_expander() -> None:
    render_expander(
        "🏆 Which model is *best*? (Depends on the decision goal.)",
        best_model_depends_on_goal_md(),
    )


def render_calibration_caveat_expander() -> None:
    render_expander(
        "🎯 Why we treat scores as resemblance, not probability",
        why_calibration_matters_md()
        + "\n\n"
        + why_probabilities_should_not_be_overinterpreted_md(),
    )
