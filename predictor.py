"""Multi-horizon gold price predictor (XGBoost when available, numpy Ridge fallback)."""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("gold_bot")

DEFAULT_DB = "gold_bot.db"
DEFAULT_MODELS_DIR = Path("models")
HORIZONS = (1, 7, 30)
FEATURE_FILE = "feature_columns.json"
METRICS_FILE = "metrics.json"
BACKEND_FILE = "backend.json"
STATUS_FILE = "training_status.json"
HISTORY_FILE = "training_history.jsonl"
LAST_PREDICT_FILE = "last_prediction.json"

try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None  # type: ignore


class RidgeRegressor:
    """Simple L2-regularized linear regressor (numpy-only fallback)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        Xs = (X - self.mean_) / self.std_
        ones = np.ones((Xs.shape[0], 1))
        Xb = np.hstack([ones, Xs])
        n_features = Xb.shape[1]
        reg = self.alpha * np.eye(n_features)
        reg[0, 0] = 0.0
        beta = np.linalg.pinv(Xb.T @ Xb + reg) @ Xb.T @ y
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.coef_ + self.intercept_


def models_ready(models_dir: Path | str = DEFAULT_MODELS_DIR) -> bool:
    models_dir = Path(models_dir)
    return all((models_dir / f"model_{h}d.pkl").exists() for h in HORIZONS) and (
        models_dir / FEATURE_FILE
    ).exists()


def _fetch_rows(db_path: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT timestamp, tala_price, usd_price, ounce_price, fair_price, difference, source
            FROM price_history
            WHERE tala_price IS NOT NULL AND usd_price IS NOT NULL AND ounce_price IS NOT NULL
            ORDER BY timestamp ASC
            """
        )
        return c.fetchall()
    finally:
        conn.close()


def _parse_ts(value: str):
    from datetime import datetime as dt

    text = str(value).strip().replace("Z", "+00:00")
    try:
        return dt.fromisoformat(text)
    except ValueError:
        return dt.strptime(text[:19], "%Y-%m-%d %H:%M:%S")


def load_daily_bars(db_path: str = DEFAULT_DB) -> dict[str, np.ndarray]:
    """Build daily OHLC bars from every price_history row (all sources).

    Uses the full DB: open/high/low/close + tick count + intraday range from
    all intraday snapshots, then trains day-ahead horizons on those bars.
    """
    rows = _fetch_rows(db_path)
    if not rows:
        return {}

    by_day: dict[str, dict[str, float]] = {}
    for ts, tala, usd, ounce, fair, diff, _src in rows:
        day = str(ts)[:10]
        tala_f = float(tala)
        usd_f = float(usd)
        ounce_f = float(ounce)
        fair_f = float(fair or 0.0)
        diff_f = float(diff or 0.0)
        if day not in by_day:
            by_day[day] = {
                "open": tala_f,
                "high": tala_f,
                "low": tala_f,
                "close": tala_f,
                "usd": usd_f,
                "ounce": ounce_f,
                "fair": fair_f,
                "difference": diff_f,
                "ticks": 1.0,
            }
        else:
            bar = by_day[day]
            bar["high"] = max(bar["high"], tala_f)
            bar["low"] = min(bar["low"], tala_f)
            bar["close"] = tala_f
            bar["usd"] = usd_f
            bar["ounce"] = ounce_f
            bar["fair"] = fair_f
            bar["difference"] = diff_f
            bar["ticks"] += 1.0

    days = sorted(by_day.keys())
    open_ = np.array([by_day[d]["open"] for d in days], dtype=float)
    high = np.array([by_day[d]["high"] for d in days], dtype=float)
    low = np.array([by_day[d]["low"] for d in days], dtype=float)
    close = np.array([by_day[d]["close"] for d in days], dtype=float)
    usd = np.array([by_day[d]["usd"] for d in days], dtype=float)
    ounce = np.array([by_day[d]["ounce"] for d in days], dtype=float)
    fair = np.array([by_day[d]["fair"] for d in days], dtype=float)
    diff = np.array([by_day[d]["difference"] for d in days], dtype=float)
    ticks = np.array([by_day[d]["ticks"] for d in days], dtype=float)
    range_pct = (high - low) / np.where(close == 0, np.nan, close)
    intraday_ret = (close - open_) / np.where(open_ == 0, np.nan, open_)

    from datetime import datetime as dt

    dow = np.array([dt.strptime(d, "%Y-%m-%d").weekday() for d in days], dtype=float)
    return {
        "days": np.array(days),
        "tala": close,  # close is the main price series
        "open": open_,
        "high": high,
        "low": low,
        "usd": usd,
        "ounce": ounce,
        "fair": fair,
        "difference": diff,
        "ticks": ticks,
        "range_pct": range_pct,
        "intraday_ret": intraday_ret,
        "dow": dow,
    }


def load_daily_arrays(db_path: str = DEFAULT_DB) -> dict[str, np.ndarray]:
    """Daily last-close series from all sources (charts / history tail)."""
    bars = load_daily_bars(db_path)
    if not bars:
        return {}
    return {
        "days": bars["days"],
        "tala": bars["tala"],
        "usd": bars["usd"],
        "ounce": bars["ounce"],
        "fair": bars["fair"],
        "difference": bars["difference"],
        "dow": bars["dow"],
    }


def load_series_arrays(db_path: str = DEFAULT_DB) -> dict[str, np.ndarray]:
    """Raw intraday rows (debug / compatibility). Prefer load_daily_bars for ML."""
    rows = _fetch_rows(db_path)
    if not rows:
        return {}

    timestamps = [_parse_ts(r[0]) for r in rows]
    tala = np.array([r[1] for r in rows], dtype=float)
    usd = np.array([r[2] for r in rows], dtype=float)
    ounce = np.array([r[3] for r in rows], dtype=float)
    fair = np.array([r[4] or 0.0 for r in rows], dtype=float)
    diff = np.array([r[5] or 0.0 for r in rows], dtype=float)
    dow = np.array([t.weekday() for t in timestamps], dtype=float)
    days = np.array([t.strftime("%Y-%m-%d") for t in timestamps])
    return {
        "timestamps": np.array(timestamps, dtype=object),
        "days": days,
        "tala": tala,
        "usd": usd,
        "ounce": ounce,
        "fair": fair,
        "difference": diff,
        "dow": dow,
    }


def _lag(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if n < len(arr):
        out[n:] = arr[:-n]
    return out


def _rolling_mean(arr: np.ndarray, win: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    for i in range(win - 1, len(arr)):
        out[i] = (csum[i + 1] - csum[i + 1 - win]) / win
    return out


def _rolling_std(arr: np.ndarray, win: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(win - 1, len(arr)):
        out[i] = np.std(arr[i + 1 - win : i + 1])
    return out


def _rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period + 1:
        return out
    delta = np.diff(arr, prepend=arr[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    for i in range(period, len(arr)):
        g = gain[i - period + 1 : i + 1].mean()
        l = loss[i - period + 1 : i + 1].mean()
        if l < 1e-12:
            out[i] = 100.0
        else:
            rs = g / l
            out[i] = 100 - (100 / (1 + rs))
    return out


FEATURE_NAMES = []
TARGET_MODE = "return"  # models predict fractional returns; inference converts to price


def build_feature_matrix(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Return X (n, f), feature names, and horizon return targets y_h."""
    global FEATURE_NAMES
    tala, usd, ounce, diff = data["tala"], data["usd"], data["ounce"], data["difference"]
    cols = []
    names = []

    def add(name, arr):
        names.append(name)
        cols.append(np.asarray(arr, dtype=float))

    for series_name, series in (("tala", tala), ("usd", usd), ("ounce", ounce), ("difference", diff)):
        add(series_name, series)
        for lag in (1, 2, 3, 7, 14):
            add(f"{series_name}_lag{lag}", _lag(series, lag))
        for win in (7, 14, 30):
            add(f"{series_name}_ma{win}", _rolling_mean(series, win))
            add(f"{series_name}_std{win}", _rolling_std(series, win))

    ret_1d = np.full_like(tala, np.nan)
    ret_1d[1:] = (tala[1:] - tala[:-1]) / np.where(tala[:-1] == 0, np.nan, tala[:-1])
    ret_7d = np.full_like(tala, np.nan)
    ret_7d[7:] = (tala[7:] - tala[:-7]) / np.where(tala[:-7] == 0, np.nan, tala[:-7])
    add("ret_1d", ret_1d)
    add("ret_7d", ret_7d)
    add("rsi_14", _rsi(tala, 14))
    add("dow", data["dow"])

    # Full-DB intraday aggregates when daily bars are available
    if "range_pct" in data:
        add("range_pct", data["range_pct"])
        add("range_pct_ma7", _rolling_mean(data["range_pct"], 7))
    if "intraday_ret" in data:
        add("intraday_ret", data["intraday_ret"])
        add("intraday_ret_ma7", _rolling_mean(data["intraday_ret"], 7))
    if "ticks" in data:
        add("ticks", data["ticks"])
        add("ticks_ma7", _rolling_mean(data["ticks"], 7))

    # Distance from moving averages (trend position)
    ma7 = _rolling_mean(tala, 7)
    ma14 = _rolling_mean(tala, 14)
    add("tala_vs_ma7", (tala - ma7) / np.where(ma7 == 0, np.nan, ma7))
    add("tala_vs_ma14", (tala - ma14) / np.where(ma14 == 0, np.nan, ma14))

    X = np.column_stack(cols)
    FEATURE_NAMES = names

    # Horizon targets as fractional returns (stable vs absolute price level)
    targets: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        y = np.full_like(tala, np.nan)
        if h < len(tala):
            future = tala[h:]
            base = tala[:-h]
            y[:-h] = (future - base) / np.where(base == 0, np.nan, base)
        targets[h] = y
    return X, names, targets


def _make_model():
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=2,
            random_state=42,
        )
    return RidgeRegressor(alpha=10.0)


def _train_one(X: np.ndarray, y: np.ndarray) -> tuple[Any, dict]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv, yv = X[mask], y[mask]
    if len(Xv) < 40:
        raise RuntimeError(f"Not enough samples to train ({len(Xv)})")

    split = int(len(Xv) * 0.8)
    X_train, X_test = Xv[:split], Xv[split:]
    y_train, y_test = yv[:split], yv[split:]

    model = _make_model()
    model.fit(X_train, y_train)

    metrics: dict[str, float] = {
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
        "backend": "xgboost" if HAS_XGBOOST else "ridge",
        "target": TARGET_MODE,
    }
    if len(X_test) > 0:
        pred = np.asarray(model.predict(X_test), dtype=float)
        # Metrics in return space; also report MAE in % points
        mae = float(np.mean(np.abs(pred - y_test)))
        metrics["mae_return"] = mae
        metrics["mae"] = mae * 100.0  # percentage points for admin display
        metrics["mape"] = mae * 100.0
        # Directional accuracy (up/down)
        metrics["dir_acc"] = float(np.mean(np.sign(pred) == np.sign(y_test)) * 100)
    return model, metrics


def _sanitize_return(pred_ret: float, horizon: int, recent_rets: np.ndarray) -> float:
    """Clamp extreme forecasts using recent realized volatility."""
    valid = recent_rets[np.isfinite(recent_rets)]
    if len(valid) < 5:
        cap = {1: 0.08, 7: 0.20, 30: 0.35}.get(horizon, 0.15)
    else:
        vol = float(np.std(valid))
        # Scale vol roughly with sqrt(horizon) for multi-day
        scale = max(1.0, float(np.sqrt(horizon)))
        cap = max(0.03 * scale, min(0.50, 3.5 * vol * scale))
    return float(np.clip(pred_ret, -cap, cap))


def _write_status(models_dir: Path, **fields):
    from datetime import datetime, timezone

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    (models_dir / STATUS_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def get_training_status(models_dir: Path | str = DEFAULT_MODELS_DIR) -> dict:
    path = Path(models_dir) / STATUS_FILE
    if not path.exists():
        return {"state": "unknown"}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"state": "unknown"}


def append_training_history(entry: dict, models_dir: Path | str = DEFAULT_MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_training_history(limit: int = 10, models_dir: Path | str = DEFAULT_MODELS_DIR) -> list[dict]:
    path = Path(models_dir) / HISTORY_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return list(reversed(entries))


def model_artifact_info(models_dir: Path | str = DEFAULT_MODELS_DIR) -> dict:
    """Filesystem + metrics snapshot for admin monitoring."""
    from datetime import datetime

    models_dir = Path(models_dir)
    info = {
        "ready": models_ready(models_dir),
        "backend": "xgboost" if HAS_XGBOOST else "ridge",
        "models_dir": str(models_dir.resolve()),
        "horizons": {},
        "feature_count": 0,
        "metrics": backtest_summary(models_dir),
        "status": get_training_status(models_dir),
        "trained_at": None,
        "daily_samples": 0,
    }
    backend_path = models_dir / BACKEND_FILE
    if backend_path.exists():
        try:
            info["backend"] = json.loads(backend_path.read_text()).get("backend", info["backend"])
        except Exception:
            pass

    feat_path = models_dir / FEATURE_FILE
    if feat_path.exists():
        try:
            info["feature_count"] = len(json.loads(feat_path.read_text()))
        except Exception:
            pass

    metrics_path = models_dir / METRICS_FILE
    if metrics_path.exists():
        info["trained_at"] = datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(timespec="seconds")

    for h in HORIZONS:
        p = models_dir / f"model_{h}d.pkl"
        info["horizons"][f"{h}d"] = {
            "exists": p.exists(),
            "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else 0,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.exists() else None,
        }

    try:
        rows = _fetch_rows()
        info["row_samples"] = len(rows)
        daily = load_daily_bars()
        info["daily_samples"] = len(daily.get("tala", [])) if daily else 0
    except Exception:
        info["row_samples"] = 0
        info["daily_samples"] = 0

    last_pred = models_dir / LAST_PREDICT_FILE
    if last_pred.exists():
        try:
            info["last_prediction"] = json.loads(last_pred.read_text())
        except Exception:
            info["last_prediction"] = None
    else:
        info["last_prediction"] = None

    return info


def save_last_prediction(prediction: dict, models_dir: Path | str = DEFAULT_MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **{k: prediction.get(k) for k in (
            "model_ready", "price_now", "pred_1d", "pred_7d", "pred_30d",
            "expected_return_1d", "expected_return_7d", "expected_return_30d", "error",
        )},
    }
    (models_dir / LAST_PREDICT_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def train_and_save(
    db_path: str = DEFAULT_DB,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    triggered_by: str = "cli",
) -> dict:
    from datetime import datetime, timezone

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    _write_status(
        models_dir,
        state="running",
        triggered_by=triggered_by,
        started_at=started.isoformat(),
        message="Training in progress",
    )

    try:
        data = load_daily_bars(db_path)
        if not data:
            raise RuntimeError("No price history available for training")

        # Count raw rows so admin UI shows full-DB usage
        row_samples = len(_fetch_rows(db_path))

        X, names, targets = build_feature_matrix(data)
        all_metrics: dict[str, Any] = {}
        for h in HORIZONS:
            model, metrics = _train_one(X, targets[h])
            with open(models_dir / f"model_{h}d.pkl", "wb") as f:
                pickle.dump(model, f)
            all_metrics[f"{h}d"] = metrics
            logger.info("Trained model_%sd: %s", h, metrics)

        finished = datetime.now(timezone.utc)
        duration_sec = (finished - started).total_seconds()
        all_metrics["_meta"] = {
            "trained_at": finished.isoformat(),
            "duration_sec": duration_sec,
            "triggered_by": triggered_by,
            "row_samples": int(row_samples),
            "daily_samples": int(len(data["tala"])),
            "feature_count": len(names),
            "backend": "xgboost" if HAS_XGBOOST else "ridge",
            "target": TARGET_MODE,
        }

        (models_dir / FEATURE_FILE).write_text(json.dumps(names, ensure_ascii=False, indent=2))
        (models_dir / METRICS_FILE).write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
        (models_dir / BACKEND_FILE).write_text(
            json.dumps(
                {
                    "backend": "xgboost" if HAS_XGBOOST else "ridge",
                    "target": TARGET_MODE,
                },
                indent=2,
            )
        )

        history_entry = {
            "trained_at": finished.isoformat(),
            "duration_sec": duration_sec,
            "triggered_by": triggered_by,
            "success": True,
            "metrics": {k: v for k, v in all_metrics.items() if k != "_meta"},
            "meta": all_metrics["_meta"],
        }
        append_training_history(history_entry, models_dir)
        _write_status(
            models_dir,
            state="idle",
            last_success_at=finished.isoformat(),
            last_duration_sec=duration_sec,
            triggered_by=triggered_by,
            message="Last training succeeded",
        )
        return all_metrics
    except Exception as e:
        finished = datetime.now(timezone.utc)
        append_training_history(
            {
                "trained_at": finished.isoformat(),
                "duration_sec": (finished - started).total_seconds(),
                "triggered_by": triggered_by,
                "success": False,
                "error": str(e),
            },
            models_dir,
        )
        _write_status(
            models_dir,
            state="error",
            last_error_at=finished.isoformat(),
            triggered_by=triggered_by,
            message=str(e),
        )
        raise


def backtest_summary(models_dir: Path | str = DEFAULT_MODELS_DIR) -> dict:
    path = Path(models_dir) / METRICS_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    # Strip meta for callers that only want horizons
    return {k: v for k, v in data.items() if k != "_meta"}


def backtest_summary_full(models_dir: Path | str = DEFAULT_MODELS_DIR) -> dict:
    path = Path(models_dir) / METRICS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def clear_models(models_dir: Path | str = DEFAULT_MODELS_DIR) -> list[str]:
    """Delete model artifacts (keeps history/status). Returns deleted filenames."""
    models_dir = Path(models_dir)
    deleted = []
    for name in (
        *[f"model_{h}d.pkl" for h in HORIZONS],
        FEATURE_FILE,
        METRICS_FILE,
        BACKEND_FILE,
        LAST_PREDICT_FILE,
    ):
        p = models_dir / name
        if p.exists():
            p.unlink()
            deleted.append(name)
    _write_status(models_dir, state="idle", message="Models cleared by admin")
    return deleted


def _load_models(models_dir: Path):
    cols = json.loads((models_dir / FEATURE_FILE).read_text())
    models = {}
    for h in HORIZONS:
        with open(models_dir / f"model_{h}d.pkl", "rb") as f:
            models[h] = pickle.load(f)
    return models, cols


def predict_future(
    db_path: str = DEFAULT_DB,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    live_tala: float | None = None,
    live_usd: float | None = None,
    live_ounce: float | None = None,
) -> dict[str, Any]:
    models_dir = Path(models_dir)
    empty = {
        "model_ready": False,
        "price_now": live_tala or 0,
        "pred_1d": 0,
        "pred_7d": 0,
        "pred_30d": 0,
        "expected_return_1d": 0.0,
        "expected_return_7d": 0.0,
        "expected_return_30d": 0.0,
        "expected_return": 0.0,
        "usd_toman": live_usd,
        "ounce": live_ounce,
        "error": "models_missing",
    }
    if not models_ready(models_dir):
        return empty

    try:
        models, cols = _load_models(models_dir)
        target_mode = TARGET_MODE
        backend_path = models_dir / BACKEND_FILE
        if backend_path.exists():
            try:
                target_mode = json.loads(backend_path.read_text()).get("target", TARGET_MODE)
            except Exception:
                pass

        data = load_daily_bars(db_path)
        if not data:
            empty["error"] = "no_history"
            return empty

        if live_tala is not None:
            data["tala"][-1] = live_tala
            if "high" in data:
                data["high"][-1] = max(float(data["high"][-1]), float(live_tala))
            if "low" in data:
                data["low"][-1] = min(float(data["low"][-1]), float(live_tala))
            if live_usd is not None:
                data["usd"][-1] = live_usd
            if live_ounce is not None:
                data["ounce"][-1] = live_ounce
            if live_usd and live_ounce:
                fair = live_usd * live_ounce / 41.5
                data["fair"][-1] = fair
                data["difference"][-1] = live_tala - fair

        X, names, _targets = build_feature_matrix(data)
        # Align to saved feature order
        name_to_idx = {n: i for i, n in enumerate(names)}
        row_vals = []
        for c in cols:
            if c not in name_to_idx:
                empty["error"] = f"missing_feature:{c}"
                return empty
            row_vals.append(X[-1, name_to_idx[c]])
        row = np.array(row_vals, dtype=float).reshape(1, -1)

        # If last row incomplete, walk backward
        if not np.all(np.isfinite(row)):
            found = False
            for i in range(len(X) - 1, -1, -1):
                candidate = np.array([X[i, name_to_idx[c]] for c in cols], dtype=float)
                if np.all(np.isfinite(candidate)):
                    row = candidate.reshape(1, -1)
                    found = True
                    break
            if not found:
                empty["error"] = "incomplete_features"
                return empty

        price_now = float(live_tala if live_tala is not None else data["tala"][-1])
        raw_preds = {h: float(np.asarray(models[h].predict(row)).ravel()[0]) for h in HORIZONS}

        # Recent 1d returns for volatility clamp
        recent_rets = np.full(len(data["tala"]), np.nan)
        tala_arr = data["tala"]
        recent_rets[1:] = (tala_arr[1:] - tala_arr[:-1]) / np.where(tala_arr[:-1] == 0, np.nan, tala_arr[:-1])
        recent_window = recent_rets[-60:]

        preds: dict[int, float] = {}
        rets: dict[int, float] = {}
        for h in HORIZONS:
            if target_mode == "return":
                ret = _sanitize_return(raw_preds[h], h, recent_window)
                rets[h] = ret
                preds[h] = price_now * (1.0 + ret)
            else:
                # Legacy absolute-price models
                preds[h] = raw_preds[h]
                rets[h] = (preds[h] - price_now) / price_now if price_now else 0.0

        result = {
            "model_ready": True,
            "price_now": price_now,
            "pred_1d": preds[1],
            "pred_7d": preds[7],
            "pred_30d": preds[30],
            "expected_return_1d": rets[1] * 100,
            "expected_return_7d": rets[7] * 100,
            "expected_return_30d": rets[30] * 100,
            "usd_toman": live_usd if live_usd is not None else float(data["usd"][-1]),
            "ounce": live_ounce if live_ounce is not None else float(data["ounce"][-1]),
            "error": None,
        }
        result["expected_return"] = result["expected_return_7d"]
        try:
            save_last_prediction(result, models_dir)
        except Exception:
            logger.debug("Could not persist last prediction", exc_info=True)
        return result
    except Exception as e:
        logger.exception("predict_future failed")
        empty["error"] = str(e)
        return empty


def daily_history_tail(db_path: str = DEFAULT_DB, days: int = 30):
    """Return list of (date_str, tala) for charting."""
    data = load_daily_arrays(db_path)
    if not data:
        return []
    days_arr = data["days"][-days:]
    tala = data["tala"][-days:]
    return list(zip(days_arr.tolist(), tala.tolist()))
