"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Model: per-trophy least-squares trend of logit(rate) against ln(1 + hours
since launch), with shrinkage toward the cross-trophy median slope, plus a
spread that combines a random walk in horizon, the fit's residual variance
and slope uncertainty. Two facts from RULES.txt drive the design:

1. Exophase refreshes a game's counts on its own schedule, in discrete
   jumps, not continuously: launch night the page was frozen (638 players,
   every rate identical) from 23:48Z to 04:00Z, then refreshed at 04:30Z to
   3,868 players with every rate but the one at 100% lower. A history that
   is perfectly flat therefore does NOT mean "rates are stable" -- it means
   "no refresh has landed yet", and a refresh is pending. For a flat
   history we apply a dilution prior (see _dilution_median): a refresh
   multiplies the tracked player base (x6 on launch night, per RULES) and
   the new cohort has earned each trophy at only a fraction k of the old
   cohort's rate, so the rate is diluted toward r * (1 + k*(G-1)) / G.
   The 100% trophy is the exception: a trophy every tracked player has
   earned is earned by newcomers just as fast, so k = 1 and it stays ~100%.

2. Over a week (the final: cut 21 Nov 11:30Z, target 26 Nov 00:00Z) the
   history is long and holds many refreshes, so the per-trophy log-time
   trend -- the same shape the house curve baseline uses, but fitted over
   the last 48 h (not 24 h), shrunk toward the global slope, and with a
   pooled residual -- is what carries the forecast. The dilution prior
   only fires when the history is flat, which the final's is not.

Quantiles are logit-normal around the median with spread s:
    q_tau = 100 / (1 + exp(-(logit(median) + z_tau * s)))
rounded to 4 decimals (half up), z = [-1.645, -0.674, 0, 0.674, 1.645].

The program never crashes on a case: any unexpected failure falls back to
the house baseline (last rate, random-walk spread), so the output is
always a valid forecast file. Pure float arithmetic, no randomness, no
network, no third-party imports: two runs write identical bytes.

Run:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, reads $MAKEFUN_INPUT_DIR else /input,
writes $MAKEFUN_OUTPUT_DIR else /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Random-walk spread of a rate's move, logit units per sqrt day. Half the
# house baseline's 0.5: the fit's residual variance below already carries
# the refresh-jump variance, so the full 0.5 would double-count it; 0.25
# remains as a floor for genuinely unforecastable drift over the horizon.
# (Synthetic final-like tests: 0.25 clearly beats 0.5, 0.0 beats 0.25 only
# in an unrealistically smooth world -- 0.25 is the honest middle.)
SIGMA_RW = 0.25
# Launch instant: the house curve baseline's origin, and the game's release day.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Trend window: fit the last this many hours of reads (house curve: 24 h).
TREND_WINDOW_HOURS = 48.0
# Dilution prior (flat history only): player-base multiplier at a refresh and
# the fraction of the old rate at which the new cohort has earned a trophy.
# Calibrated from RULES.txt's own published house-baseline practice numbers
# (loss ~1.51, typical miss ~0.68 points), NOT from any answer: with mean
# last rate 0.84% over the 54 non-100% trophies, a typical miss of 0.68
# points implies the refreshed rate is ~1/5 of the old one, and a mean
# |logit shift| of ~1.51 implies the same (odds ratio e^-1.51 = 0.22).
# Launch night the base grew 638 -> 3,868 (x6.06), so a refresh multiplies
# the base by ~6 while newcomers have earned almost nothing yet:
# f = (1 + k*(G-1)) / G = 0.217 for k = 0.06 below.
REFRESH_GROWTH = 6.0
NEW_COHORT_FRACTION = 0.06
# Extra logit spread for the unknown size of a pending refresh jump.
JUMP_SIGMA = 0.6


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(at: str) -> float:
    return (parse_time(at) - LAUNCH).total_seconds() / 3600.0


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y ~ a + b*x. Returns (a, b, residual variance)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        resid = 0.0
    return a, b, resid


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _dilution_median(last_rate: float) -> float:
    """Rate after one pending refresh, by dilution of the tracked base.

    Old cohort E earners among P players (rate r). After the refresh P' = G*P
    players; the G-1 cohorts of newcomers have earned the trophy at k*r.
    New rate = r * (1 + k*(G-1)) / G. The 100% trophy (k = 1) is unchanged.
    """
    if last_rate >= 99.99:
        return 99.99
    k = NEW_COHORT_FRACTION
    return clamp_pct(last_rate * (1.0 + k * (REFRESH_GROWTH - 1.0)) / REFRESH_GROWTH)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = case["target"]
    x_target = math.log1p(max(0.0, hours_since_launch(target)))

    last = history[-1]
    n_tr = len(trophies)

    # Per-read x values (log-time since launch).
    xs_all = [math.log1p(max(0.0, hours_since_launch(r["at"]))) for r in history]
    # Trend window: last 48 h of reads, else everything.
    # (window computed on hours, not log-time)
    hrs_all = [hours_since_launch(r["at"]) for r in history]
    h_last = hrs_all[-1]
    win_idx = [i for i, h in enumerate(hrs_all) if h_last - h <= TREND_WINDOW_HOURS]
    if len(win_idx) < 3:
        win_idx = list(range(len(history)))

    # Is the history perfectly flat? (No refresh has landed yet.)
    flat = all(r["rates"] == history[0]["rates"] for r in history)

    # First pass: per-trophy OLS on the window.
    fits: list[tuple[float, float, float, int, float]] = []  # (a, b, resid, n, x_mean)
    slopes: list[float] = []
    for i in range(n_tr):
        pts = [(xs_all[j], logit(history[j]["rates"][i]))
               for j in win_idx if history[j]["rates"][i] is not None]
        if len(pts) >= 3 and len({p[0] for p in pts}) >= 3:
            a, b, resid = _ols([p[0] for p in pts], [p[1] for p in pts])
            xm = sum(p[0] for p in pts) / len(pts)
            fits.append((a, b, resid, len(pts), xm))
            slopes.append(b)
        else:
            fits.append((0.0, 0.0, 0.0, len(pts), 0.0))
    global_slope = _median(slopes) if slopes else 0.0
    # Pooled residual variance across trophies with a real fit.
    resids = [f[2] for f in fits if f[3] >= 3 and f[2] > 0]
    pooled_resid = _median(resids) if resids else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            # Fall back to the most recent non-null read for this trophy.
            last_rate = next(
                (r["rates"][i] for r in reversed(history) if r["rates"][i] is not None),
                1.0,
            )
        a, b, resid, n, xm = fits[i]
        if flat:
            # No refresh has landed: the trend is uninformative (slope 0 by
            # construction). Forecast the dilution of one pending refresh.
            med = _dilution_median(last_rate)
            med_logit = logit(med)
            spread = math.sqrt(SIGMA_RW ** 2 * horizon_days + JUMP_SIGMA ** 2)
        elif n >= 3:
            # Shrink the per-trophy slope toward the global slope; refit the
            # intercept at the window's mean x with the shrunk slope.
            w = n / (n + 4.0)
            b_s = w * b + (1.0 - w) * global_slope
            ys = [logit(history[j]["rates"][i]) for j in win_idx
                  if history[j]["rates"][i] is not None]
            xs = [xs_all[j] for j in win_idx if history[j]["rates"][i] is not None]
            a_s = sum(ys) / len(ys) - b_s * (sum(xs) / len(xs))
            med_logit = a_s + b_s * x_target
            # Spread: random walk + residual (own blended with pooled) +
            # slope uncertainty growing with extrapolation distance.
            resid_eff = 0.5 * resid + 0.5 * pooled_resid
            dx = x_target - xm
            slope_var = resid_eff / max(1.0, sum((x - xm) ** 2 for x in xs)) if xs else 0.0
            spread = math.sqrt(
                SIGMA_RW ** 2 * horizon_days
                + resid_eff * (1.0 + 1.0 / max(1, n))
                + slope_var * dx * dx
            )
            spread = min(max(spread, 0.20), 2.5)
        else:
            # Too few points: persistence at the last rate, global slope only.
            med_logit = logit(last_rate) + global_slope * max(0.0, x_target - xs_all[-1])
            spread = math.sqrt(SIGMA_RW ** 2 * horizon_days + pooled_resid + 0.25)
            spread = min(max(spread, 0.30), 2.5)

        if last_rate >= 99.99:
            # The trophy every tracked player has earned: newcomers earn it
            # just as fast, so it stays ~100%; keep a narrow downward spread.
            med_logit = logit(99.99)
            spread = min(spread, 0.30)
        med_logit = max(-9.21, min(9.21, med_logit))
        rows.append({
            "id": trophy["id"],
            "q": [round4(inv_logit(med_logit + z * spread)) for z in Z],
        })
    return {"trophies": rows}


def baseline_case(case: dict) -> dict:
    """House baseline fallback: last rate, random-walk spread. Never raises
    on a well-formed case; used if the model above fails for any reason."""
    last = case["history"][-1] if case["history"] else None
    spread = SIGMA_RW * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if last is not None else None
        if rate is None:
            rate = 1.0
        lg = logit(rate)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = baseline_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
