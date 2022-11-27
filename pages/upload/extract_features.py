from datetime import date

import pandas as pd
from pandas import DataFrame, Series
from pandas.errors import ParserError
import streamlit as st


def extract_ts_col(df: DataFrame) -> tuple[str, str]:
    good_cols, bad_cols = [], []
    for col in df.columns:
        try:
            if df[col].dtype == "int64":
                tscol = _parse_s(df[col])
                if _check_tscol_int(tscol):
                    good_cols.append((col, "s"))

                tscol = _parse_ms(df[col])
                if _check_tscol_int(tscol):
                    good_cols.append((col, "ms"))

            else:
                _ = _parse_str(df[col])
                good_cols.append((col, "str"))
        except ParserError as e:
            bad_cols.append(col)
            st.exception(e)
        except ValueError as e:
            bad_cols.append(col)
            st.exception(e)
    st.write(good_cols, bad_cols)
    return good_cols[0]


def _parse_s(df_col: Series):
    st.write(df_col)
    st.write(pd.to_datetime(df_col, unit="s"))
    return pd.to_datetime(df_col, unit="s")


def _parse_ms(df_col: Series):
    st.write(df_col)
    return pd.to_datetime(df_col, unit="ms")


def _parse_str(df_col: Series):
    return pd.to_datetime(df_col)


def _check_tscol_int(ts_col: Series):
    min_ts, max_ts = ts_col.min(), ts_col.max()
    return min_ts > date(1990, 1, 1) and max_ts < date(3000, 1, 1)


def _check_tscol_str(ts_col: Series):
    return True
