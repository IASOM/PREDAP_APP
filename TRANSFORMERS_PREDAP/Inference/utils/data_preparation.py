"""Data preparation helpers used by the inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from utils.experiments_utils import smart_read

pd.read_csv = smart_read

MAX_DATE = "2027-09-30"


def split_train_test(df, split_ratio=0.8):
    split_idx = int(len(df) * split_ratio)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def normalize_dataframe(train_df, test_df, csv_file=None, save_data=False, target_code=None, scaler=None):
    if "timestamp" not in train_df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")

    if scaler is None:
        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
    else:
        scaler_target = clone(scaler)

    codes = [code for code in train_df.columns if code not in {"timestamp", target_code}]
    scaler.fit(train_df[codes].values)
    train_df[codes] = scaler.transform(train_df[codes].values)
    test_df[codes] = scaler.transform(test_df[codes].values)

    if target_code is not None:
        scaler_target = scaler_target.fit(train_df[[target_code]].values)
        test_df[[target_code]] = scaler_target.transform(test_df[[target_code]].values)
        train_df[[target_code]] = scaler_target.transform(train_df[[target_code]].values)

    if save_data and csv_file is not None:
        input_path = Path(csv_file)
        output_file = input_path.parent / f"train_df_normalized_{input_path.name}"
        train_df.to_csv(output_file, index=False)
        print(f"Normalized data saved to: {output_file}")

    return train_df, test_df


def cut_dataframe(df: pd.DataFrame, date_cutoff: str = "2010-01-01", max_date: str = MAX_DATE, csv_file: str = None, save_data: bool = False) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cutoff = pd.Timestamp(date_cutoff)
    max_dt = pd.Timestamp(max_date)
    df = df[(df["timestamp"] >= cutoff) & (df["timestamp"] <= max_dt)].reset_index(drop=True)

    if save_data and csv_file is not None:
        input_path = Path(csv_file)
        output_file = input_path.parent / f"date_{date_cutoff}_{input_path.name}"
        df.to_csv(output_file, index=False)
        print(f"Cut data saved to: {output_file}")

    return df


def eliminate_covid_dates(df: pd.DataFrame, covid_periods: list) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for start_date, end_date in covid_periods:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[~((df["timestamp"] >= start) & (df["timestamp"] <= end))].reset_index(drop=True)
    return df


def add_covid_token(df):
    df = df.copy()
    waves = {
        "Primera Onada": ("2020-03-01", "2020-06-30"),
        "Segona Onada": ("2020-10-01", "2020-12-31"),
        "Tercera Onada": ("2021-01-01", "2021-03-31"),
        "Quarta Onada": ("2021-04-01", "2021-06-30"),
        "Cinquena Onada": ("2021-07-01", "2021-09-30"),
    }

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["covid_token"] = 0
    for _, (start, end) in waves.items():
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        df.loc[df["timestamp"].between(start_date, end_date), "covid_token"] = 1
    return df


def prepare_time_series_features(df, categorical_vars, cutoff_date="2010-01-01", max_date: str = MAX_DATE, scaler=None, eliminate_covid_data=False, covid_dates=None):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if eliminate_covid_data:
        if covid_dates is None:
            raise ValueError("covid_dates must be provided when eliminate_covid_data=True")
        df = eliminate_covid_dates(df, covid_dates)

    cutoff = pd.Timestamp(cutoff_date)
    max_dt = pd.Timestamp(max_date)
    df = df[(df["timestamp"] >= cutoff) & (df["timestamp"] <= max_dt)].reset_index(drop=True)

    if scaler is None:
        scaler = MinMaxScaler()

    fixed_holidays = {
        "New Year's Day": "01-01",
        "Epiphany": "01-06",
        "Labour Day": "05-01",
        "Feast of St. John the Baptist": "06-24",
        "Assumption of the Virgin": "08-15",
        "National Day of Catalonia": "09-11",
        "Hispanic Day": "10-12",
        "All Saints' Day": "11-01",
        "Constitution Day": "12-06",
        "Immaculate Conception Day": "12-08",
        "Christmas Day": "12-25",
        "St. Stephen's Day": "12-26",
    }

    def get_movable_holidays(year):
        good_friday = easter(year) - pd.Timedelta(days=2)
        easter_monday = easter(year) + pd.Timedelta(days=1)
        return {"Good Friday": good_friday, "Easter Monday": easter_monday}

    public_holidays = []
    for year in range(df["timestamp"].min().year, df["timestamp"].max().year + 1):
        for holiday_name, date_str in fixed_holidays.items():
            public_holidays.append((pd.Timestamp(f"{year}-{date_str}"), holiday_name))
        for holiday_name, holiday_date in get_movable_holidays(year).items():
            public_holidays.append((holiday_date, holiday_name))

    public_holidays_df = pd.DataFrame(public_holidays, columns=["timestamp", "Holiday"])
    public_holidays_df["timestamp"] = pd.to_datetime(public_holidays_df["timestamp"])

    date_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="D")
    df_dates = pd.DataFrame(
        {
            "timestamp": date_range,
            "Day_of_Week": date_range.day_name(),
            "Month": date_range.month_name(),
            "Season": date_range.month.map(
                lambda m: "Winter" if m in [12, 1, 2] else "Spring" if m in [3, 4, 5] else "Summer" if m in [6, 7, 8] else "Autumn"
            ),
        }
    )

    dow = df_dates["timestamp"].dt.dayofweek.astype(float)
    month = (df_dates["timestamp"].dt.month - 1).astype(float)
    doy = (df_dates["timestamp"].dt.dayofyear - 1)
    days_in_year = np.where(df_dates["timestamp"].dt.is_leap_year, 366.0, 365.0).astype(float)

    df_dates["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df_dates["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df_dates["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df_dates["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df_dates["doy_sin"] = np.sin(2 * np.pi * doy / days_in_year)
    df_dates["doy_cos"] = np.cos(2 * np.pi * doy / days_in_year)

    df_dates = df_dates.drop(columns=["Day_of_Week", "Month", "Season"], errors="ignore")
    categorical_vars = [var for var in categorical_vars if var not in ["Day_of_Week", "Month", "Season"]]

    df_dates["Is_Weekend"] = (df_dates["timestamp"].dt.dayofweek >= 5).astype(bool)
    df_dates = df_dates.merge(public_holidays_df, on="timestamp", how="left")
    df_dates["Holiday"] = df_dates["Holiday"].fillna("No Holiday")

    school_vacations = [
        ("Christmas Break", "12-23", "01-07"),
        ("Easter Break", "03-23", "04-06"),
        ("Summer Break", "06-22", "09-10"),
    ]

    def get_school_vacation(date):
        year = date.year
        for vacation_name, start_date, end_date in school_vacations:
            if start_date.startswith("12") and end_date.startswith("01"):
                start_dec = pd.Timestamp(f"{year - 1}-{start_date}")
                end_jan = pd.Timestamp(f"{year}-{end_date}")
                if start_dec <= date <= end_jan:
                    return vacation_name
            else:
                start = pd.Timestamp(f"{year}-{start_date}")
                end = pd.Timestamp(f"{year}-{end_date}")
                if start <= date <= end:
                    return vacation_name
        return "No Vacation"

    df_dates["School_Vacation"] = df_dates["timestamp"].apply(get_school_vacation)
    df_dates["Holiday"] = np.where(df_dates["Holiday"] != "No Holiday", "Holiday", "No_Holiday")
    df_dates["School_Vacation"] = np.where(df_dates["School_Vacation"] != "No Vacation", "Vacation", "No_Vacation")

    df_final = pd.get_dummies(df_dates, columns=categorical_vars, drop_first=False)
    if eliminate_covid_data:
        df_final = eliminate_covid_dates(df_final, covid_dates)
    return df_final
