import numpy as np
import pandas as pd
import pytest

from forecaster.data.household import load_minutes, to_hourly


def _write(tmp_path, rows):
    path = tmp_path / "power.txt"
    lines = ["Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3"]
    lines += [f"{d};{t};{p};0;0;0;0;0;0" for d, t, p in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_question_marks_become_missing(tmp_path):
    rows = [("16/12/2006", "17:24:00", "4.216"), ("16/12/2006", "17:25:00", "?")]
    series = load_minutes(_write(tmp_path, rows))
    assert series.iloc[0] == pytest.approx(4.216)
    assert np.isnan(series.iloc[1])


def test_timestamps_are_parsed_day_first(tmp_path):
    series = load_minutes(_write(tmp_path, [("02/01/2007", "00:00:00", "1.0")]))
    assert series.index[0] == pd.Timestamp("2007-01-02 00:00:00")


def test_partial_hours_are_blanked(tmp_path):
    minutes = [(f"01/01/2007", f"00:{m:02d}:00", "2.0") for m in range(40)]
    minutes += [("01/01/2007", f"01:{m:02d}:00", "3.0") for m in range(10)]
    hourly = to_hourly(load_minutes(_write(tmp_path, minutes)), min_valid_minutes=30)
    assert hourly["kw"].iloc[0] == pytest.approx(2.0)
    assert np.isnan(hourly["kw"].iloc[1])
    assert hourly["valid_minutes"].tolist() == [40, 10]


def test_invalid_min_valid_minutes_raises(tmp_path):
    minutes = load_minutes(_write(tmp_path, [("01/01/2007", "00:00:00", "1.0")]))
    with pytest.raises(ValueError):
        to_hourly(minutes, min_valid_minutes=0)
