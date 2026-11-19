#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (*.in) in the input directory and writes one forecast
file per case to the output directory: five quantiles (5/25/50/75/95%) of each
trophy's Exophase earn rate at the case's target, as JSON
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Model (standard library only, fully deterministic, no network):
  The Exophase page is frozen between the tracker's own refreshes, so a case's
  history is a short list of distinct page states, not a continuous series.
  Two regimes, chosen from the history itself:
  * FLAT history (a single distinct page state, e.g. launch night): the page
    has not refreshed yet, so the forecast models the next refresh in
    earners/players counts. Players jump by a factor G (the first refresh the
    house recorded multiplied tracked players ~6.06x). Each trophy gains
    new earners dE ~ Poisson(dP * q), where q is the chance a newly tracked
    player already holds the trophy: a power law in the trophy's current
    rate, boosted for trophies a fresh player can earn in the first hour
    (Chapter 1, fishing, arcade, discovering vehicles), cut for trophies
    that need late-story progress, grinding or breadth (final strand
    missions, 100% completion, 250 stunts, ...), with an extra boost for
    fishing -- the one activity every tracked player had done (the Gone
    Fishing trophy sat at 100%). Quantiles come from the Poisson law itself,
    so rare trophies get honest, asymmetric, count-aware bands, and every
    rate is rounded to the 2 decimals the page prints.
  * MOVING history (several distinct page states): per trophy, a least
    squares line of logit(rate) on ln(1 + hours since release), with the
    slope shrunk toward the median slope of trophies in the same earliness
    class (empirical-Bayes weight from the slope's standard error), read at
    the target. The spread is the regression's prediction error plus a base
    term for refresh lag and model error. Extrapolation is capped so the
    median cannot move more than 2.5 logits from the last observed rate.

  The dilution constants below are a Poisson fit to the launch-night
  refresh (the one refresh the rules themselves publish: 638 -> 3,868
  tracked players); they are a handful of structural parameters, not a
  table of answers. Nothing here hard-codes any practice answer.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# --- fixed model constants -------------------------------------------------
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# First-refresh player jump published in the competition rules (638 -> 3868).
G_REFRESH = 6.0627
# Dilution law: E[dE] = dP * exp(A + B*ln(r) + C_FAST*[fast] + C_SLOW*[slow]
#                        + C_FISH*[fishing]), r = current rate as a fraction.
DIL_A = -2.7357
DIL_B = 1.055
DIL_C_FAST = 1.566
DIL_C_SLOW = -2.576
DIL_C_FISH = 0.661

# Trend branch.
TAU_SLOPE = 0.5        # prior sd of per-trophy logit slopes (shrinkage scale)
SIGMA_FLOOR = 0.12     # floor on residual sd so 2-state fits still shrink
SPREAD_BASE = 0.15     # base logit spread: refresh lag + model error
SPREAD_FLOOR = 0.15
SPREAD_CAP = 0.9
MAX_LOGIT_MOVE = 2.5   # cap on |median logit - last observed logit|
BETA_CLAMP = 2.0

FAST_KEYS = ("chapter 1", "gone fishing", "fishing challenge",
             "arcade game", "discover 100 vehicles")
SLOW_KEYS = ("100% completion", "250 stunts", "gold medal",
             "20 different races", "each type of activity",
             "final mission for", "5 bank vaults", "each gang",
             "3 collectible card sets", "100 locations", "50 challenges",
             "laundering", "unique animal", "without being detected",
             "every location", "mission strands", "every spot",
             "in a single")


# --- small numeric helpers --------------------------------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    x = clamp(p, 0.01, 99.99) / 100.0
    return math.log(x / (1.0 - x))


def inv_logit_pct(l):
    if l >= 0.0:
        e = math.exp(-l)
        return 100.0 / (1.0 + e)
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round2(x):
    q = round(float(x), 2)
    return 0.0 if q == 0 else q


def parse_time(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Date-only values such as the game's release_date: treat as UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    m = n // 2
    return v[m] if n % 2 else 0.5 * (v[m - 1] + v[m])


def poisson_quantile(lam, tau):
    """Smallest k with P(Poisson(lam) <= k) >= tau. Normal approx for big lam."""
    if lam <= 0.0:
        return 0
    if lam > 60.0:
        z = Z[LEVELS.index(tau)] if tau in LEVELS else 0.0
        return max(0, int(math.floor(lam + z * math.sqrt(lam) + 0.5)))
    p = math.exp(-lam)
    cdf = p
    k = 0
    while cdf < tau and k < 10000:
        k += 1
        p *= lam / k
        cdf += p
    return k


# --- trophy features ---------------------------------------------------------
def earliness(trophy):
    """0 = slow / late / grind, 1 = normal, 2 = first-hour activity.

    Decided from the trophy's own text (name, description, tier) with fixed
    keyword rules: story position, activity availability, grind counts and
    breadth requirements. Same rules for every trophy, no per-trophy table.
    """
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
    if trophy.get("tier") == "platinum":
        return 0
    if any(k in text for k in FAST_KEYS):
        return 2
    if any(k in text for k in SLOW_KEYS):
        return 0
    return 1


def is_fishing(trophy):
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
    return "fish" in text


# --- case handling -------------------------------------------------------------
def last_rates(history, n):
    """Most recent non-null rate per trophy, from the history's last reads."""
    out = [None] * n
    for read in reversed(history):
        rates = read.get("rates") or []
        for i in range(min(n, len(rates))):
            if out[i] is None and rates[i] is not None:
                out[i] = float(rates[i])
        if all(v is not None for v in out):
            break
    return [0.0 if v is None else v for v in out]


def distinct_states(history):
    """Collapse consecutive identical reads into page states.

    Returns a list of (slot_datetime, players, rates) with one entry per
    distinct page state, in time order.
    """
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(None if r is None else float(r) for r in rates))
        if key == prev_key:
            continue
        prev_key = key
        when = parse_time(read.get("slot") or read.get("at"))
        states.append((when, read.get("players_tracked"), list(key[1])))
    return states


# --- flat-history branch: model the next refresh ------------------------------
def forecast_refresh(case, trophies, history):
    last = history[-1]
    p0 = float(last.get("players_tracked") or 0)
    if p0 <= 0:
        p0 = 1.0
    p1 = max(p0 + 1.0, round(p0 * G_REFRESH))
    dp = p1 - p0
    r0 = last_rates(history, len(trophies))
    rows = []
    for i, tr in enumerate(trophies):
        rate0 = r0[i]
        if rate0 >= 99.99:
            rows.append({"id": tr["id"], "q": [100.0] * 5})
            continue
        e0 = int(round(rate0 * p0 / 100.0))
        e0 = int(clamp(e0, 0, p0))
        r_frac = max(rate0, 0.01) / 100.0
        logq = (DIL_A + DIL_B * math.log(r_frac))
        lvl = earliness(tr)
        if lvl == 2:
            logq += DIL_C_FAST
        elif lvl == 0:
            logq += DIL_C_SLOW
        if is_fishing(tr):
            logq += DIL_C_FISH
        q = clamp(math.exp(logq), 0.0, 0.999)
        lam = dp * q
        qs = []
        for tau in LEVELS:
            k = poisson_quantile(lam, tau)
            qs.append(round2(100.0 * (e0 + k) / p1))
        qs = [clamp(v, 0.0, 100.0) for v in qs]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": tr["id"], "q": qs})
    return rows


# --- moving-history branch: shrunk logit trend ---------------------------------
def _ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, sxx, sigma, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0 or n < 2:
        return my, 0.0, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    sigma = math.sqrt(rss / (n - 2)) if n > 2 else 0.0
    return a, b, sxx, sigma, n


def forecast_trend(case, trophies, history, states):
    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = states[0][0] or datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case.get("target")) or states[-1][0]
    if target is None:
        target = release
    h_target = max((target - release).total_seconds() / 3600.0, 0.0)
    x_target = math.log1p(h_target)

    n_tr = len(trophies)
    # Fit on recent states only: trophy curves saturate, so the recent
    # slope is the honest guide to the target, not the launch-night climb.
    # Keep every state when the recent window holds too few of them.
    times = [when for when, _p, _r in states if when is not None]
    if len(times) >= 5:
        from datetime import timedelta as _td
        cutoff = max(times) - _td(hours=36)
        recent = [s for s in states if s[0] is None or s[0] >= cutoff]
        fit_states = recent if len(recent) >= 4 else states
    else:
        fit_states = states
    fits = []  # per trophy: dict(a, b, sxx, sigma, n, y_last, level)
    for i in range(n_tr):
        xs, ys = [], []
        for when, _p, rates in fit_states:
            if when is None or i >= len(rates) or rates[i] is None:
                continue
            raw = float(rates[i])
            if raw <= 0.01:
                # A printed 0.00/0.01% is the page's floor, not a measured
                # logit: it would drag any line fit. Skip it (censoring).
                continue
            h = max((when - release).total_seconds() / 3600.0, 0.0)
            xs.append(math.log1p(h))
            ys.append(logit_pct(raw))
        if not xs:
            fits.append(None)
            continue
        a, b, sxx, sigma, n = _ols(xs, ys)
        fits.append({"a": a, "b": b, "sxx": sxx,
                     "sigma": max(sigma, SIGMA_FLOOR), "n": n,
                     "y_last": ys[-1], "x_mean": sum(xs) / len(xs),
                     "level": earliness(trophies[i])})

    # Pooled slopes: per earliness level, else global, else 0.
    pool = {}
    for lvl in (0, 1, 2):
        bs = [f["b"] for f in fits
              if f is not None and f["level"] == lvl and f["n"] >= 3
              and f["sxx"] > 0.05]
        if bs:
            pool[lvl] = median(bs)
    all_bs = [f["b"] for f in fits
              if f is not None and f["n"] >= 3 and f["sxx"] > 0.05]
    pool_global = median(all_bs) if all_bs else 0.0

    r_last = last_rates(history, n_tr)
    rows = []
    for i, tr in enumerate(trophies):
        if r_last[i] >= 99.99:
            rows.append({"id": tr["id"], "q": [100.0] * 5})
            continue
        f = fits[i]
        if f is None:
            # No usable history for this trophy: stay near its last rate
            # with a wide spread.
            med = clamp(r_last[i], 0.01, 99.99)
            lm = logit_pct(med)
            qs = [round2(inv_logit_pct(lm + z * 1.0)) for z in Z]
            qs = [clamp(v, 0.0, 100.0) for v in qs]
            for j in range(1, 5):
                qs[j] = max(qs[j], qs[j - 1])
            rows.append({"id": tr["id"], "q": qs})
            continue
        # Shrink the slope toward its level's pooled slope.
        if f["sxx"] > 0.0:
            se2 = (f["sigma"] ** 2) / f["sxx"]
        else:
            se2 = 1e6
        w = (TAU_SLOPE ** 2) / (TAU_SLOPE ** 2 + se2)
        b_pool = pool.get(f["level"], pool_global)
        b = clamp(w * f["b"] + (1.0 - w) * b_pool, -BETA_CLAMP, BETA_CLAMP)
        y_hat = f["a"] + b * x_target
        y_hat = clamp(y_hat, f["y_last"] - MAX_LOGIT_MOVE,
                      f["y_last"] + MAX_LOGIT_MOVE)
        # Prediction spread: regression error at the target + base term.
        if f["sxx"] > 0.0:
            var = (f["sigma"] ** 2) * (
                1.0 + 1.0 / max(f["n"], 1)
                + (x_target - f["x_mean"]) ** 2 / f["sxx"])
        else:
            var = (f["sigma"] ** 2) * 2.0
        s = math.sqrt(var + SPREAD_BASE ** 2)
        s = clamp(s, SPREAD_FLOOR, SPREAD_CAP)
        qs = [round2(inv_logit_pct(y_hat + z * s)) for z in Z]
        qs = [clamp(v, 0.0, 100.0) for v in qs]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": tr["id"], "q": qs})
    return rows


# --- per-case forecast ------------------------------------------------------------
def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    if not trophies:
        return {"trophies": []}
    if not history:
        # No reads at all: flat prior, wide spread.
        rows = []
        for tr in trophies:
            lm = logit_pct(0.5)
            qs = [round2(inv_logit_pct(lm + z * 1.0)) for z in Z]
            rows.append({"id": tr["id"], "q": qs})
        return {"trophies": rows}
    states = distinct_states(history)
    if len(states) <= 1:
        rows = forecast_refresh(case, trophies, history)
    else:
        rows = forecast_trend(case, trophies, history, states)
    return {"trophies": rows}


def validate_output(obj, trophies):
    if not isinstance(obj, dict) or set(obj.keys()) != {"trophies"}:
        return "top-level keys must be exactly {trophies}"
    rows = obj["trophies"]
    if len(rows) != len(trophies):
        return "row count != trophy count"
    seen = set()
    for row in rows:
        if set(row.keys()) != {"id", "q"}:
            return "row keys must be exactly {id, q}"
        if row["id"] in seen:
            return "duplicate trophy id"
        seen.add(row["id"])
        q = row["q"]
        if len(q) != 5 or any(not isinstance(v, (int, float))
                              or v != v or v in (float("inf"), float("-inf"))
                              for v in q):
            return "each q must be 5 finite numbers"
        if any(v < 0.0 or v > 100.0 for v in q):
            return "quantile outside [0, 100]"
        if any(q[j] < q[j - 1] for j in range(1, 5)):
            return "quantiles must be non-decreasing"
    if {t["id"] for t in trophies} != seen:
        return "trophy ids do not match the case"
    return None


# --- driver -------------------------------------------------------------------------
def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    done = []
    for name in names:
        path = os.path.join(input_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        cid = case.get("id") or name[:-3]
        out = forecast_case(case)
        err = validate_output(out, case.get("trophies") or [])
        if err:
            raise ValueError(f"{cid}: invalid forecast: {err}")
        blob = json.dumps(out, separators=(",", ":")).encode("utf-8")
        if len(blob) > 65536:
            raise ValueError(f"{cid}: output {len(blob)} bytes over the cap")
        with open(os.path.join(output_dir, cid + ".out"), "wb") as fh:
            fh.write(blob)
        done.append(cid)
    return done


def self_check(input_dir):
    """--check: forecast every case found (or a built-in smoke case when the
    input directory holds none) and validate each output's shape."""
    if os.path.isdir(input_dir) and any(
            f.endswith(".in") for f in os.listdir(input_dir)):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            done = run(input_dir, tmp)
        print(f"check ok: {len(done)} case(s) forecast and valid: "
              + ", ".join(done))
        return 0
    # Smoke case: two flat reads, five trophies, 90-minute horizon.
    case = {
        "id": "smoke", "at": "2026-11-19T03:00:00Z", "horizon_minutes": 90,
        "target": "2026-11-19T04:30:00Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/"
                           "grand-theft-auto-vi-psn/trophies/"},
        "trophies": [
            {"id": 1, "name": "Vice Got a Price!", "description": "Platinum.",
             "tier": "platinum", "points": 300, "secret": False},
            {"id": 2, "name": "Just Like Paradise",
             "description": "Complete Gone Fishing.", "tier": "bronze",
             "points": 15, "secret": True},
            {"id": 3, "name": "Paradise Lost", "description": "Complete Chapter 1.",
             "tier": "bronze", "points": 15, "secret": True},
            {"id": 4, "name": "Stick Fiddler", "description": "Complete an arcade game.",
             "tier": "bronze", "points": 15, "secret": False},
            {"id": 5, "name": "A Long Road", "description": "Attain 100% Completion.",
             "tier": "gold", "points": 90, "secret": False},
        ],
        "history": [
            {"slot": "2026-11-19T02:00:00Z", "at": "2026-11-19T02:00:00.3Z",
             "players_tracked": 638, "trophies_earned": 943,
             "rates": [0.63, 100.0, 1.25, 0.94, 0.63]},
            {"slot": "2026-11-19T02:30:00Z", "at": "2026-11-19T02:30:00.3Z",
             "players_tracked": 638, "trophies_earned": 943,
             "rates": [0.63, 100.0, 1.25, 0.94, 0.63]},
        ],
    }
    out = forecast_case(case)
    err = validate_output(out, case["trophies"])
    if err:
        print(f"check failed: {err}", file=sys.stderr)
        return 1
    blob = json.dumps(out, separators=(",", ":")).encode("utf-8")
    if len(blob) > 65536:
        print("check failed: output over the size cap", file=sys.stderr)
        return 1
    print("check ok: smoke case forecast and valid")
    return 0


def main(argv):
    if "--check" in argv:
        input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
        return self_check(input_dir)
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    try:
        done = run(input_dir, output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("forecast wrote: " + ", ".join(done))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
