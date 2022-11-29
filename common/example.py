from functools import partial

import streamlit as st
import pandas as pd
import altair as alt

from pages.adjust.adjustments import adjust_datetime, adjust_long, adjust_resample


def run():
    path = '~/Downloads/download.csv'
    df = get_test_data(path)
    st.session_state['df'] = df
    df["variable"] = ""
    df["variable"] = df["variable"].str.cat(df[["node", "metric_type"]], sep="-")
    chart = alt.Chart(df[df["variable"].isin(['-node1-login', '-node2-login'])].reset_index()).mark_line().encode(
        x='_time:T',
        y='value:Q',
        color='variable:N'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    adjusts = [
        partial(adjust_datetime, "_time", "ms"),
        partial(adjust_long, ["node", "metric_type"], "value", "dt"),
        partial(adjust_resample, "1H", "max")
    ]

    for adj in adjusts:
        df = adj(df)
    st.session_state['df_clear'] = df
    chart = alt.Chart(df.reset_index()).transform_fold(
        ['-node1-login', '-node2-login']
    ).mark_line().encode(
        x='dt:T',
        y='value:Q',
        color='key:N'
    )
    st.altair_chart(chart, use_container_width=True)


@st.experimental_memo(ttl=3600)
def get_test_data(path: str):
    return pd.read_csv(path)
