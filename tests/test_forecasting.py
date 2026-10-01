import pandas as pd
import pytest

from forecast_lab.core import BacktestConfig, backtest, generate_demo, scores, validate_data


def test_backtest_respects_time_boundary_and_computes_metrics():
    results = backtest(generate_demo(days=220), BacktestConfig(180, 14, 14))
    assert set(results.model) == {"last_value", "same_weekday", "ridge_calendar"}
    assert (results.date > results.cutoff).all()
    assert (results.forecast >= 0).all()
    summary = scores(results)
    assert summary.shape[0] == 9
    assert summary.wape_pct.notna().all()


def test_future_actuals_cannot_affect_earlier_fold_predictions():
    data = generate_demo(days=220)
    config = BacktestConfig(180, 14, 14)
    original = backtest(data, config)
    changed = data.copy()
    first_cutoff = original.cutoff.min()
    # Change actuals inside the first forecast window, leaving all training rows intact.
    changed.loc[changed.date > first_cutoff, "units"] = 999
    rerun = backtest(changed, config)
    left = original[original.cutoff == first_cutoff].forecast.reset_index(drop=True)
    right = rerun[rerun.cutoff == first_cutoff].forecast.reset_index(drop=True)
    pd.testing.assert_series_equal(left, right)


def test_invalid_data_rejected():
    data = generate_demo(days=100)
    with pytest.raises(ValueError, match="duplicate"):
        validate_data(pd.concat([data, data.iloc[[0]]]))
    with pytest.raises(ValueError, match="too little history"):
        backtest(data)
    with pytest.raises(ValueError, match="whole numbers"):
        validate_data(data.assign(units=1.5))
    with pytest.raises(ValueError, match="at least 7"):
        backtest(data, BacktestConfig(6, 14, 14))
