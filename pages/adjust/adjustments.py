import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype


def adjust_datetime(col_name: str, date_format: str, df: DataFrame):
    df["dt"] = (
        pd.to_datetime(df[col_name], unit=date_format)
        if df[col_name].dtype == 'int64'
        else pd.to_datetime(df[col_name], format=date_format)
    )
    df = df.drop(col_name, axis=1)
    return df


def adjust_long(variable_col: list[str], value_col: str, date_col: str, df: DataFrame):
    df['variable'] = df[variable_col[0]]
    df["variable"] = df["variable"].str.cat(df[variable_col[1:]], sep="-")
    time_col = 'dt' if 'dt' in df.columns else date_col
    return df.pivot(index=time_col, columns="variable", values=value_col).reset_index()


def adjust_resample(freq: str, func: str, df: DataFrame):
    cols_to_resample = [c for c in df.columns if is_numeric_dtype(df[c])]
    return df.set_index('dt').resample(freq)[cols_to_resample].agg(func)
