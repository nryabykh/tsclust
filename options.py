import streamlit as st
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List


@dataclass
class DataOptions:
    date: date
    eps: float
    bm_list: List[str]


def get_options() -> DataOptions:
    bm_selected = st.multiselect(
        label='Select metrics',
        options=data.get_bm_list(),
        default=data.get_bm_list(),
    )

    col_date, col_margin, col_slider, col_right = st.columns((1, 0.2, 1, 1))

    # dates = defaults.get_history_dates()
    start_date = date.today() - timedelta(days=30)

    selected_start_date = col_date.date_input(
        label='BM history started at',
        value=start_date,
    )

    help_text = ("DBSCAN algorithm is used for clustering. You can change the '_epsilon_' parameter of DBSCAN with "
                 "this slider. You could consider '_epsilon_' as maximum allowed distance between the objects of the"
                 "same cluster. Mutual Pearson correlations subtracted from 1 are taken as distances between BMs. "
                 "More details about DBSCAN you can find there: "
                 "https://scikit-learn.org/stable/modules/clustering.html#dbscan")

    selected_eps = col_slider.slider(
        'Number of clusters adjustment',
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help=help_text
    )

    return DataOptions(date=selected_start_date, eps=selected_eps, bm_list=bm_selected)