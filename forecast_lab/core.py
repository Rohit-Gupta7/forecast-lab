"""Synthetic demand, forecast models, and rolling-origin evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED = {"date", "sku", "units", "promo"}


def generate_demo(days: int = 420, seed: int = 7) -> pd.DataFrame:
    """Create fictional daily sales with weekly patterns, trends and promotions."""
    if days < 90:
        raise ValueError("days must be at least 90")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    frames = []
    for sku, base, trend, weekend_lift in [
        ("coffee", 48, 0.045, 16), ("notebook", 32, 0.018, -7), ("headphones", 20, 0.034, 9)
    ]:
        t = np.arange(days)
        promo = ((t % 31) >= 26).astype(int)
        weekend = (dates.dayofweek >= 5).astype(int)
        units = np.maximum(0, np.rint(base + trend * t + weekend_lift * weekend
                                       + 11 * promo + rng.normal(0, 3.5, days))).astype(int)
        frames.append(pd.DataFrame({"date": dates, "sku": sku, "units": units, "promo": promo}))
    return pd.concat(frames, ignore_index=True)


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"data must contain: {', '.join(sorted(REQUIRED))}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data[["date", "sku", "units", "promo"]].isna().any().any():
        raise ValueError("required columns cannot contain missing values")
    if data.duplicated(["sku", "date"]).any():
        raise ValueError("duplicate sku/date rows")
    if (data["units"] < 0).any() or not np.isfinite(data["units"]).all():
        raise ValueError("units must be finite and nonnegative")
    if not data["promo"].isin([0, 1]).all():
        raise ValueError("promo must be 0 or 1")
    for sku, group in data.groupby("sku"):
        dates = pd.DatetimeIndex(group["date"].sort_values())
        if len(dates) > 1 and not dates.to_series().diff().iloc[1:].dt.days.eq(1).all():
            raise ValueError(f"{sku} has missing calendar dates")
    return data.sort_values(["sku", "date"]).reset_index(drop=True)


def features(data: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """Known-at-forecast-time inputs only: calendar position and planned promo."""
    day = pd.to_datetime(data["date"])
    weekday = pd.get_dummies(day.dt.dayofweek).reindex(columns=range(7), fill_value=0)
    t = (day - origin).dt.days.to_numpy()
    return np.column_stack([t, weekday.to_numpy(), data["promo"].to_numpy()])


def predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    if model == "last_value":
        return np.repeat(float(train["units"].iloc[-1]), len(test))
    if model == "same_weekday":
        history = train.tail(7).set_index(train.tail(7)["date"].dt.dayofweek)["units"]
        return np.array([float(history.loc[day]) for day in test["date"].dt.dayofweek])
    if model == "ridge_calendar":
        origin = train["date"].iloc[0]
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        estimator.fit(features(train, origin), train["units"].to_numpy())
        return np.maximum(0, estimator.predict(features(test, origin)))
    raise ValueError(f"unknown model: {model}")


@dataclass(frozen=True)
class BacktestConfig:
    min_train_days: int = 180
    horizon_days: int = 28
    step_days: int = 28


def backtest(frame: pd.DataFrame, config: BacktestConfig = BacktestConfig()) -> pd.DataFrame:
    """Evaluate entire future windows with training restricted to previous dates."""
    if min(config.min_train_days, config.horizon_days, config.step_days) <= 0:
        raise ValueError("backtest windows must be positive")
    data = validate_data(frame)
    outputs = []
    for sku, group in data.groupby("sku", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < config.min_train_days + config.horizon_days:
            raise ValueError(f"{sku} has too little history for one fold")
        for cut in range(config.min_train_days, len(group) - config.horizon_days + 1, config.step_days):
            train = group.iloc[:cut]
            test = group.iloc[cut:cut + config.horizon_days]
            assert train["date"].max() < test["date"].min()
            for name in ("last_value", "same_weekday", "ridge_calendar"):
                forecast = predict(train, test, name)
                for date, actual, prediction in zip(test["date"], test["units"], forecast):
                    outputs.append({"sku": sku, "cutoff": train["date"].max(),
                                    "date": date, "model": name, "actual": int(actual),
                                    "forecast": round(float(prediction), 3)})
    return pd.DataFrame(outputs)


def scores(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.assign(error=lambda x: x.forecast - x.actual,
                              abs_error=lambda x: (x.forecast - x.actual).abs())
    results = data.groupby(["sku", "model"], as_index=False).agg(
        mae=("abs_error", "mean"), absolute_error=("abs_error", "sum"),
        actual_units=("actual", "sum"), bias=("error", "mean"), forecasts=("forecast", "size"))
    results["wape_pct"] = np.where(results.actual_units > 0,
                                   100 * results.absolute_error / results.actual_units, np.nan)
    return results.drop(columns=["absolute_error", "actual_units"]).round(3)
