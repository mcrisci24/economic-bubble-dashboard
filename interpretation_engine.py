"""Centralized interpretation logic for the dashboard.

This module is the single source of truth for:

1. **Regime badge classification** — translating warning/recovery scores and
   the rule-based phase label into one of the six user-facing badges.
2. **Signal confirmation table** — combining the rule-based monitor, the
   supervised ML risk score, and the hazard/onset score into a per-ticker
   confluence view.
3. **Research story construction** — the dynamic interpretation rendered on
   the Executive Summary page. It only reads values that the pipeline
   actually produced; it never invents results.

Keeping this logic out of `dashboard.py` makes it testable and prevents
inline string heuristics from drifting between pages.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard_helpers import fmt_num, fmt_pct


# ---------------------------------------------------------------------------
# Regime badge classifier
# ---------------------------------------------------------------------------
# Thresholds are deliberately simple. Their job is to convert a continuous
# rule-based state into a categorical label a non-expert can scan. They are
# not optimised; treat them as a labelling convention, not a model output.
def classify_regime_badge(
    *,
    warning_score: float | None,
    recovery_score: float | None,
    drawdown_pct: float | None,
    distance_from_200dma: float | None,
    weekly_rsi: float | None,
    phase_label: str | None = None,
) -> str:
    """Return one of: Normal, Heating Up, Euphoria / Stretched, Breakdown, Capitulation, Recovery Watch.

    Logic, in priority order:
        1. Capitulation — drawdown deeper than ~35% with oversold RSI.
        2. Breakdown — drawdown deepening below ~10% with negative momentum.
        3. Recovery Watch — recovery score elevated and drawdown stabilising.
        4. Euphoria / Stretched — high warning score *and* stretched momentum.
        5. Heating Up — moderate warning score or stretched-but-not-extreme.
        6. Normal — none of the above.
    """
    dd = _safe_float(drawdown_pct)
    dist = _safe_float(distance_from_200dma)
    rsi = _safe_float(weekly_rsi)
    warn = _safe_float(warning_score)
    rec = _safe_float(recovery_score)
    phase = (phase_label or "").lower()

    # 1. Capitulation
    if dd is not None and dd <= -35 and (rsi is None or rsi <= 35):
        return "Capitulation"
    if "capitulation" in phase:
        return "Capitulation"

    # 2. Breakdown
    if dd is not None and dd <= -10 and (dist is None or dist <= 0):
        return "Breakdown"
    if "breakdown" in phase or "crash" in phase:
        return "Breakdown"

    # 3. Recovery Watch — only when drawdown has stabilised
    if rec is not None and rec >= 45 and (dd is None or dd >= -25):
        return "Recovery Watch"
    if "recovery" in phase:
        return "Recovery Watch"

    # 4. Euphoria / Stretched
    extreme_momentum = (rsi is not None and rsi >= 75) or (dist is not None and dist >= 30)
    extreme_warn = warn is not None and warn >= 65
    if extreme_warn and extreme_momentum:
        return "Euphoria / Stretched"
    if "euphoria" in phase or "stretched" in phase:
        return "Euphoria / Stretched"

    # 5. Heating Up
    moderate_warn = warn is not None and warn >= 40
    moderate_momentum = (rsi is not None and rsi >= 65) or (dist is not None and dist >= 15)
    if moderate_warn or moderate_momentum:
        return "Heating Up"

    return "Normal"


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal confirmation
# ---------------------------------------------------------------------------
def build_signal_confirmation_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine rule-based, ML, and hazard signals into one per-ticker view.

    The table intentionally avoids saying "buy" or "sell." It reports whether
    independent research signals point in the same direction, and produces a
    `signal_confirmation_level` of None / Low / Medium / High based on how
    many signals are simultaneously elevated.
    """
    ai = data.get("current_ai_ml_risk", pd.DataFrame()).copy()
    hazard = data.get("hazard_current_ai", pd.DataFrame()).copy()
    monitor = data.get("ai_monitor", pd.DataFrame()).copy()

    if ai.empty and hazard.empty and monitor.empty:
        return pd.DataFrame()

    tickers: set[str] = set()
    for df in (ai, hazard, monitor):
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].dropna().astype(str).tolist())

    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        row: dict[str, Any] = {"ticker": ticker}
        _attach_ml_signal(row, ai, ticker)
        _attach_hazard_signal(row, hazard, ticker)
        _attach_monitor_signal(row, monitor, ticker)
        _score_confluence(row)
        # Compute regime badge from the monitor row when available.
        row["regime_badge"] = classify_regime_badge(
            warning_score=row.get("warning_score"),
            recovery_score=row.get("recovery_score"),
            drawdown_pct=row.get("drawdown_pct"),
            distance_from_200dma=row.get("distance_from_200dma"),
            weekly_rsi=row.get("weekly_rsi") if "weekly_rsi" in row else None,
            phase_label=row.get("phase_label") or row.get("signal_label"),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_ml_signal(row: dict[str, Any], ai: pd.DataFrame, ticker: str) -> None:
    if ai.empty or "ticker" not in ai.columns:
        return
    a = ai[ai["ticker"].astype(str).eq(ticker)].tail(1)
    if a.empty:
        return
    ar = a.iloc[0]
    row["ml_model"] = ar.get("primary_model", ar.get("model", "n/a"))
    row["ml_scope"] = ar.get("primary_model_scope", ar.get("model_scope", "n/a"))
    row["ml_raw_score"] = ar.get("primary_ml_risk_probability", np.nan)
    row["ml_empirical_bucket_rate"] = ar.get("calibrated_empirical_risk", np.nan)
    row["ml_risk_category"] = ar.get("risk_category", "n/a")


def _attach_hazard_signal(row: dict[str, Any], hazard: pd.DataFrame, ticker: str) -> None:
    if hazard.empty or "ticker" not in hazard.columns:
        return
    h = hazard[hazard["ticker"].astype(str).eq(ticker)].tail(1)
    if h.empty:
        return
    hr = h.iloc[0]
    row["hazard_model"] = hr.get("hazard_model", "n/a")
    row["hazard_feature_set"] = hr.get("hazard_feature_set", "n/a")
    row["hazard_score"] = hr.get("hazard_score_4w", hr.get("hazard_probability_4w", np.nan))
    row["hazard_risk_category"] = hr.get("hazard_risk_category", "n/a")


def _attach_monitor_signal(row: dict[str, Any], monitor: pd.DataFrame, ticker: str) -> None:
    if monitor.empty or "ticker" not in monitor.columns:
        return
    m = monitor[monitor["ticker"].astype(str).eq(ticker)].tail(1)
    if m.empty:
        return
    mr = m.iloc[0]
    for c in ["phase_label", "signal_label", "warning_score", "recovery_score",
              "drawdown_pct", "distance_from_200dma", "weekly_rsi"]:
        if c in m.columns:
            row[c] = mr.get(c)


def _score_confluence(row: dict[str, Any]) -> None:
    """Fill in confluence score + level + human-readable reasons."""
    score = 0
    reasons: list[str] = []
    ml_cat = str(row.get("ml_risk_category", "")).lower()
    hazard_cat = str(row.get("hazard_risk_category", "")).lower()
    phase = str(row.get("phase_label", row.get("signal_label", ""))).lower()
    warning = pd.to_numeric(row.get("warning_score", np.nan), errors="coerce")

    if any(x in ml_cat for x in ["high", "extreme", "elevated"]):
        score += 1
        reasons.append("ML risk bucket elevated")
    if any(x in hazard_cat for x in ["high", "extreme", "elevated"]):
        score += 1
        reasons.append("Hazard/onset score elevated")
    if pd.notna(warning) and warning >= 2:
        score += 1
        reasons.append("Rule warning score active")
    if any(x in phase for x in ["euphoria", "breakdown", "risk", "crash"]):
        score += 1
        reasons.append("Rule phase label is risk-oriented")

    row["signal_confirmation_score"] = score
    # `pd.cut` of a single value is awkward; use plain branching.
    if score <= 0:
        level = "None"
    elif score == 1:
        level = "Low"
    elif score == 2:
        level = "Medium"
    else:
        level = "High"
    row["signal_confirmation_level"] = level
    row["confirmation_reasons"] = "; ".join(reasons) if reasons else "No strong cross-signal confirmation"


# ---------------------------------------------------------------------------
# Research story builder
# ---------------------------------------------------------------------------
def build_research_story(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Construct the executive-summary narrative from existing artifacts.

    Each bullet is gated on the presence of the underlying file. If
    `python rare_event_analysis.py` has not been run, the dashboard simply
    omits the rare-event bullet rather than inventing one.
    """
    story: dict[str, Any] = {
        "headline": "No complete model story is available yet.",
        "bullets": [],
        "tables": {},
    }

    results = data.get("ml_results", pd.DataFrame()).copy()
    top_decile = data.get("ml_top_decile", pd.DataFrame()).copy()
    top_k = data.get("ml_top_k_lift", pd.DataFrame()).copy()
    imbalance = data.get("imbalance_results", pd.DataFrame()).copy()
    ai = data.get("current_ai_ml_risk", pd.DataFrame()).copy()
    hazard_current = data.get("hazard_current_ai", pd.DataFrame()).copy()
    hazard_audit = data.get("hazard_audit", pd.DataFrame()).copy()
    poisson = data.get("poisson_results", pd.DataFrame()).copy()
    cost = data.get("cost_threshold_results", pd.DataFrame()).copy()
    importance = data.get("ml_feature_importance", pd.DataFrame()).copy()

    bullets: list[str] = []
    tables = story["tables"]

    # 1. Best PR-AUC model on test split
    if not results.empty and "split" in results.columns:
        test = results[results["split"].eq("test")].copy()
        if not test.empty and "pr_auc" in test.columns:
            best_pr = test.sort_values("pr_auc", ascending=False).iloc[0]
            scope = best_pr.get("asset_segment", best_pr.get("model_scope", "unknown scope"))
            bullets.append(
                f"The strongest ordinary test-set classifier by PR-AUC is **{best_pr['model']} / {scope}** "
                f"with PR-AUC {fmt_num(best_pr.get('pr_auc'), 3)}, ROC-AUC {fmt_num(best_pr.get('roc_auc'), 3)}, "
                f"precision {fmt_pct(best_pr.get('precision'))}, and recall {fmt_pct(best_pr.get('recall'))}."
            )
        if not test.empty and {"model", "asset_segment", "false_positive", "true_positive"}.issubset(test.columns):
            fp_table = test.copy()
            fp_table["false_positives_per_true_positive"] = fp_table.apply(
                lambda r: (r["false_positive"] / r["true_positive"]) if r["true_positive"] else np.inf,
                axis=1,
            )
            clean = fp_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["false_positives_per_true_positive"])
            if not clean.empty:
                best_clean = clean.sort_values("false_positives_per_true_positive").iloc[0]
                scope = best_clean.get("asset_segment", best_clean.get("model_scope", "unknown"))
                bullets.append(
                    f"The cleanest alert burden among test models is **{best_clean['model']} / {scope}**, "
                    f"with about {fmt_num(best_clean['false_positives_per_true_positive'], 2)} false positives "
                    "per true positive. High recall alone can be fake-good when the model floods the dashboard with alerts."
                )

    # 2. Top-decile lift
    if not top_decile.empty and "split" in top_decile.columns:
        td_test = top_decile[top_decile["split"].eq("test")].copy()
        if not td_test.empty:
            best_td = td_test.sort_values("lift_vs_base_rate", ascending=False).iloc[0]
            bullets.append(
                f"The clearest risk-ranking evidence comes from top-risk buckets: "
                f"**{best_td['model']} / {best_td['model_scope']}** has a top-decile event rate of "
                f"{fmt_pct(best_td.get('top_decile_event_rate'))} versus a base rate of "
                f"{fmt_pct(best_td.get('base_event_rate'))}, a lift of "
                f"{fmt_num(best_td.get('lift_vs_base_rate'), 2)}×. That is risk enrichment, not certainty."
            )
            tables["Top-decile leaders"] = td_test.sort_values("lift_vs_base_rate", ascending=False).head(8)

    # 3. Top-k lift
    if not top_k.empty and "split" in top_k.columns:
        tk_test = top_k[top_k["split"].eq("test")].copy()
        if not tk_test.empty:
            best_tk = tk_test.sort_values("lift_vs_base_rate", ascending=False).iloc[0]
            bullets.append(
                f"The highest top-k lift is **{best_tk['model']} / {best_tk['model_scope']} / {best_tk['top_label']}**, "
                f"with event rate {fmt_pct(best_tk.get('top_event_rate'))} versus base "
                f"{fmt_pct(best_tk.get('base_event_rate'))} — {fmt_num(best_tk.get('lift_vs_base_rate'), 2)}× lift. "
                "This supports treating the model as a ranking tool."
            )
            tables["Top-k lift leaders"] = tk_test.sort_values("lift_vs_base_rate", ascending=False).head(8)

    # 4. Imbalance experiments
    if not imbalance.empty and "split" in imbalance.columns:
        imb_test = imbalance[imbalance["split"].eq("test")].copy()
        if not imb_test.empty:
            best_imb = imb_test.sort_values("pr_auc", ascending=False).iloc[0]
            burden = best_imb.get("false_positives_per_true_positive", np.nan)
            bullets.append(
                f"Imbalance experiments help but are not magic. The best by PR-AUC is "
                f"**{best_imb['model']} / {best_imb['model_scope']}** with PR-AUC "
                f"{fmt_num(best_imb.get('pr_auc'), 3)}, precision {fmt_pct(best_imb.get('precision'))}, "
                f"recall {fmt_pct(best_imb.get('recall'))}, FP/TP burden {fmt_num(burden, 2)}. "
                "Keep a method only if lift improves without alert flooding."
            )
            tables["Imbalance method leaders"] = imb_test.sort_values("pr_auc", ascending=False).head(8)

    # 5. Current AI ranking
    if not ai.empty and "calibrated_empirical_risk" in ai.columns:
        ai_sorted = ai.sort_values("calibrated_empirical_risk", ascending=False)
        leader = ai_sorted.iloc[0]
        bullets.append(
            f"Among current AI-exposed assets, the highest empirical bucket risk is "
            f"**{leader['ticker']}** at {fmt_pct(leader.get('calibrated_empirical_risk'))}. "
            f"Its raw model score is {fmt_pct(leader.get('primary_ml_risk_probability'))}, but emphasise the "
            "empirical bucket rate because calibration is imperfect."
        )
        keep_cols = [
            "ticker", "asset_segment", "warning_score", "drawdown_pct", "distance_from_200dma",
            "weekly_rsi", "primary_model", "primary_model_scope", "primary_ml_risk_probability",
            "calibrated_empirical_risk", "above_tuned_threshold", "risk_category",
        ]
        tables["Current AI empirical-risk ranking"] = ai_sorted[
            [c for c in keep_cols if c in ai_sorted.columns]
        ].head(12)
        if "warning_score" in ai.columns:
            hot_cols = [c for c in ["ticker", "warning_score", "distance_from_200dma", "weekly_rsi", "drawdown_pct"] if c in ai.columns]
            tables["Current technical warning leaders"] = ai.sort_values("warning_score", ascending=False).head(5)[hot_cols]

    # 6. Hazard leakage
    if not hazard_audit.empty and "audit_warning" in hazard_audit.columns:
        suspicious = hazard_audit[hazard_audit["audit_warning"].astype(str).str.contains("SUSPICIOUS", na=False)]
        if not suspicious.empty:
            bullets.append(
                "The hazard model audit shows a large drop from full features to leakage-restricted features. "
                "The full hazard model is probably detecting current drawdown-state variables; "
                "treat the **restricted** hazard model as the safer research signal."
            )
            tables["Hazard leakage audit"] = hazard_audit
    if not hazard_current.empty and "hazard_score_4w" in hazard_current.columns:
        tables["Current restricted hazard leaders"] = hazard_current.sort_values("hazard_score_4w", ascending=False).head(8)

    # 7. Poisson
    if not poisson.empty and {"split", "mean_actual_count", "mean_predicted_count"}.issubset(poisson.columns):
        ptest = poisson[poisson["split"].eq("test")]
        if not ptest.empty:
            r = ptest.iloc[0]
            bullets.append(
                f"The Poisson count model is conceptually useful but currently weak: test mean actual count is "
                f"{fmt_num(r.get('mean_actual_count'), 2)}, mean predicted count is "
                f"{fmt_num(r.get('mean_predicted_count'), 2)}, RMSE {fmt_num(r.get('rmse'), 2)}. "
                "Treat it as a stress-count experiment, not a reliable forecast."
            )
            tables["Poisson count model summary"] = poisson

    # 8. Cost-threshold tradeoff
    if not cost.empty and {"precision", "recall"}.issubset(cost.columns):
        clean_cost = cost.replace([np.inf, -np.inf], np.nan).dropna(subset=["precision", "recall"])
        if not clean_cost.empty:
            precise = clean_cost.sort_values("precision", ascending=False).iloc[0]
            bullets.append(
                f"Cost-threshold stress tests confirm the tradeoff: the cleanest high-precision alerts reach "
                f"precision {fmt_pct(precise.get('precision'))}, but recall is only "
                f"{fmt_pct(precise.get('recall'))}. Clean alerts miss many events; broad alerts create noise."
            )
            tables["High-precision cost-threshold examples"] = clean_cost.sort_values("precision", ascending=False).head(8)

    # 9. Feature importance
    if not importance.empty and {"feature", "importance"}.issubset(importance.columns):
        imp = importance.sort_values("importance", ascending=False).head(10)
        top_features = ", ".join(imp["feature"].astype(str).head(5).tolist())
        bullets.append(
            "Feature importance shows the model story is a mix of valuation, trend, volatility, and macro "
            f"stress rather than one magic variable. Top fields include: {top_features}."
        )
        tables["Top feature-importance rows"] = imp

    if bullets:
        story["headline"] = (
            "The dashboard shows a coherent but modest story: the models are weak as exact crash predictors, "
            "but useful as risk-ranking and signal-confirmation tools when top-risk buckets, false-positive "
            "burden, and leakage checks are read together."
        )
        story["bullets"] = bullets
    return story
