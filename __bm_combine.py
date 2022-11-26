"""
Simple time series clustering based on Pearson correlation:
1. Calculate pair-wise Pearson correlation coefficients between time series.
2. Apply clustering algorithm (DBSCAN) using correlation matrix as distance matrix.
3. MDS is used for visualization of time series as points on 2D-plot.

TODO. Add sample data as csv.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List, Tuple, Optional

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from streamlit.delta_generator import DeltaGenerator

import defaults
import styler
from common.multiline_chart import create_multiline_chart
from data import get_conf, otp, data
import streamlit as st
import altair as alt

data_path = get_conf("config.yaml")["data"]

TOKEN_DEFAULTS = {
    "btn_start_clustering": False,
    "btn_setup_transformations": False,
    "actions": {},
    "selected_cluster_row": {},
    "preprocessing": {},
}


@dataclass
class DataOptions:
    date: date
    eps: float
    bm_list: List[str]


class ClusterTransformAction(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    KEEP = "keep original"
    ONE = "select one"


@dataclass
class ClusterTransform:
    cluster_label: int
    func: ClusterTransformAction
    new_name: str


class StorageLevel(str, Enum):
    SESSION_STATE = "session_state"
    DISK = "disk"


def set_session_state():
    for token, default_value in TOKEN_DEFAULTS.items():
        if token not in st.session_state:
            st.session_state[token] = default_value


def set_btn_clicked(btn_name: str):
    st.session_state[btn_name] = True


def show_caption() -> None:
    st.caption(
        "Some of business metrics might be very similar to each other. In this case it's better to reduce their "
        "number by combining similar metrics to the new one. It will likely accelerate models fitting and "
        "prediction. For the reducing number of metrics, we could, for example, group similar metrics and keep only "
        "one metric per cluster. "
    )


def get_options() -> DataOptions:
    bm_selected = st.multiselect(
        label='Select metrics',
        options=data.get_bm_list(),
        default=data.get_bm_list(),
    )

    col_date, col_margin, col_slider, col_right = st.columns((1, 0.2, 1, 1))
    dates = defaults.get_history_dates()
    start_date = dates[0] if dates else None

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


@st.experimental_memo(ttl=86400)
def load_bms(start_date: date, bm_list: Optional[List[str]] = None) -> pd.DataFrame:
    if bm_list:
        filter_cmd = "".join([f"""{"search (" if i == 0 else " OR "}metric={m}""" for i, m in enumerate(bm_list)]) + ")"
    else:
        filter_cmd = "___"
    selected_ts = int(start_date.strftime("%s"))

    # Always resample to 1h minimum
    resample_cmd = """
        | eval hour = strftime(_time, "%Y-%m-%d %H")
        | stats max(value) as value by hour, metric
        | eval _time = strptime(hour, "%Y-%m-%d %H")"""
    query = f"""  
    | {data_path['bm']}  
    | where _time>={selected_ts}  
    | {filter_cmd}
    | {resample_cmd}
    | sort _time    
    | rename metric as metric_name    
    | fields _time, metric_name, value"""
    return otp.get_data(query)


@st.experimental_memo(ttl=3600)
def get_correlations(df: pd.DataFrame) -> pd.DataFrame:
    bm_pivot = df.pivot_table(
        index='dt',
        columns='metric_name',
        values='value',
        aggfunc='max')

    bm_pivot.fillna(bm_pivot.mean(), inplace=True)
    corr = bm_pivot.corr()
    return corr.fillna(0)


@st.experimental_memo(ttl=3600)
def get_clusters(df_dist: DataFrame, eps: float) -> Tuple[DataFrame, List[int]]:
    dbscan = DBSCAN(eps=eps, min_samples=3, metric='precomputed').fit(df_dist)
    labels = dbscan.labels_
    df_clusters = (
        df_dist.assign(label=labels)
        .reset_index()
        .groupby("label")["metric_name"]
        .apply(list)
        .reset_index()
    )

    if "actions" in st.session_state:
        del st.session_state["actions"]

    st.session_state["actions"] = {}
    set_actions(labels)
    return df_clusters, labels


def set_actions(labels: List[int]) -> None:
    for label in labels:
        func, new_name = (
            (ClusterTransformAction.SUM, f"sum_cl_{label}")
            if label != -1
            else (ClusterTransformAction.KEEP, "")
        )
        st.session_state["actions"][label] = ClusterTransform(label, func, new_name)


@st.experimental_memo(ttl=3600)
def get_mds(df_dist: DataFrame) -> DataFrame:
    """
    Transforms each BM to the point on 2d-space by applying multidimensional scaling.
    MDS keeps distances specified in distance matrix 'df_dist'.
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1543)
    df_mds = DataFrame(mds.fit_transform(df_dist), columns={"x", "y"})
    df_mds["name"] = df_dist.index
    return df_mds


def render_clusters_plot(df_mds: DataFrame, place: DeltaGenerator) -> None:
    place.subheader("BMs projection to the plane")
    place.caption(("Each point on the plane represents the business metric. Coordinates of points are calculated "
                   "using the MDS (multidimensional scaling) algorithm. This algorithms projects objects to the "
                   "plane keeping specified distances between objects. Mutual Pearson correlations subtracted from 1 "
                   "are taken as distances between BMs. This scatter plot helps estimating appropriate number of "
                   "clusters."))

    chart = alt.Chart(df_mds).mark_circle().encode(
        x=alt.X("x:Q", title=None),
        y=alt.Y("y:Q", title=None, scale=alt.Scale(nice=True)),
        color="cluster:N",
        size=alt.value(100),
        tooltip=["name"]
    ).properties(width=500, height=300)
    place.write(chart)


def render_cluster_table(df_clusters: pd.DataFrame, place: DeltaGenerator) -> dict:
    place.markdown("#### Clusters table")
    place.caption(("In this table You can find business metrics grouped by clusters. Click on the row to show charts "
                   "of all BMs of the cluster, as well as to specify setting for BM aggregation."))

    df_clusters["agg_type"] = df_clusters["label"].apply(lambda x: st.session_state["actions"][x].func.value)
    df_clusters["agg_name"] = df_clusters["label"].apply(lambda x: st.session_state["actions"][x].new_name)
    df_clusters["metric_name"] = df_clusters["metric_name"].apply(lambda x: "; ".join(x))

    formatter = {
        "label": ("Cluster", {"pinned": "left", "width": "10"}),
        "agg_type": ("Action", {}),
        "agg_name": ("New name", {}),
        "metric_name": ("BMs", {}),
    }

    with place:
        table = styler.draw_grid(
            df_clusters, formatter=formatter, selection="single", wrap_text=True,
            auto_height=True
        )

    return table


def render_cluster_chart(df_bm: pd.DataFrame, place: DeltaGenerator) -> None:
    place.subheader("Cluster BMs")

    selected_cluster_row = st.session_state["selected_cluster_row"]
    if not selected_cluster_row:
        place.info("Select a cluster for chart investigation")
        return

    cluster_label = selected_cluster_row["label"]
    cluster_metrics: List[str] = selected_cluster_row["metric_name"].split("; ")
    filtered_df = df_bm[df_bm["metric_name"].isin(cluster_metrics)]
    resampled_df = filtered_df.groupby("metric_name").resample("1D")["value"].max()
    resampled_df = resampled_df.reset_index().rename({"dt": "_time", "metric_name": "variable"}, axis=1)

    place.markdown("**{} {}**".format("Cluster", cluster_label))
    chart = create_multiline_chart(resampled_df, one_line_one_row=False, height=250, width=550, points=False,
                                   color_scheme="category20b")
    place.altair_chart(chart)


def render_transformer_settings(place: DeltaGenerator) -> None:
    selected_cluster_row = st.session_state["selected_cluster_row"]
    if not selected_cluster_row:
        return

    place.subheader('Aggregation')
    place.caption("Specify aggregation settings for BMs of the selected cluster")
    selected_cluster = selected_cluster_row["label"]
    selected_metrics = selected_cluster_row["metric_name"]
    selected_cluster_transform: ClusterTransform = st.session_state["actions"][selected_cluster]

    funcs = [ct for ct in ClusterTransformAction]
    default_ix = funcs.index(selected_cluster_transform.func.value)
    default_new_name = selected_cluster_transform.new_name

    func = place.selectbox(
        label="{} {}".format("Action for cluster", selected_cluster),
        options=funcs,
        index=default_ix,
        disabled=selected_cluster == -1,
        format_func=lambda x: x.value,
        key=f"selected_action_{selected_cluster}",
    )
    if func in [ClusterTransformAction.SUM, ClusterTransformAction.MEAN]:
        new_name = place.text_input("New metric name", value=f"{default_new_name}",
                                    key=f"selected_new_name_{selected_cluster}")
    elif func == ClusterTransformAction.ONE:
        new_name = place.selectbox("Select a metric", options=selected_metrics.split("; "),
                                   key=f"selected_metric_{selected_cluster}")
    else:
        new_name = ""
    btn_submit = place.button(_("Save"))
    if btn_submit:
        st.session_state["actions"][selected_cluster] = ClusterTransform(selected_cluster, func, new_name)
        st.experimental_rerun()


def transform_bm(df_bm: DataFrame, df_clusters: DataFrame) -> DataFrame:
    dfs = []
    for cluster in df_clusters.to_dict(orient="records"):
        func = cluster["agg_type"]
        name = cluster["agg_name"]
        metrics = cluster["metric_name"].split("; ")

        df_filtered = df_bm[
            df_bm["metric_name"].isin(metrics)
        ]

        if func in ["sum", "mean"]:
            df_transformed = df_filtered.groupby("dt").agg({"value": func})
            df_transformed["metric_name"] = name
        elif func == "select one":
            single_filter = df_filtered["metric_name"] == name
            df_transformed = df_filtered[single_filter]
        elif func == "keep original":
            df_transformed = df_filtered
        else:
            df_transformed = DataFrame(columns=["dt", "metric_name", "value"])

        dfs.append(df_transformed)

    return pd.concat(dfs)


def persist(
        df: DataFrame,
        level: StorageLevel = StorageLevel.SESSION_STATE,
        path: str = "./storage/preprocessing/bm_selected.csv"
) -> None:
    if level == StorageLevel.DISK:
        try:
            df.to_csv(path)
            st.success("Preprocessed business metrics successfully saved")
        except Exception as e:
            st.error(f"Failed to save prepprocessed dataframe: {e}")
    elif level == StorageLevel.SESSION_STATE:
        st.session_state["preprocessing"]["bm"] = df
        st.success("Preprocessed business metrics successfully saved")
    else:
        return


def render_transformed(df: DataFrame, col_num: int = 3) -> None:
    resampled_df = df.groupby("metric_name").resample("1D")["value"].max().reset_index()

    st.subheader("Transformed business metrics")
    chart = alt.Chart(resampled_df).mark_line().encode(
        x=alt.X("dt:T", title=None),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("metric_name:N", scale=alt.Scale(scheme="category20b"))
    ).properties(
        width=round(1000 / col_num),
        height=130
    ).facet(
        "metric_name:N",
        align="all",
        columns=3
    ).properties(
        title=''
    ).configure_header(
        labelColor='white',
    ).resolve_axis(
        x='independent',
        y='independent'
    ).resolve_scale(
        x='independent',
        y='independent'
    )
    st.altair_chart(chart)


def app():
    st.header("Business metrics preprocessing")

    set_session_state()
    show_caption()

    data_opts: DataOptions = get_options()

    btn_start_clustering = st.button(
        "Make clustering",
        on_click=set_btn_clicked, kwargs={"btn_name": "btn_start_clustering"}
    )

    if not (btn_start_clustering or st.session_state["btn_start_clustering"]):
        return

    df_bm = load_bms(start_date=data_opts.date, bm_list=data_opts.bm_list)
    df_corr = get_correlations(df_bm)
    df_dist = 1 - df_corr
    df_clusters, labels = get_clusters(df_dist, eps=data_opts.eps)
    df_mds = get_mds(df_dist)
    df_mds["cluster"] = labels

    col_plot, col_cluster_chart = st.columns(2)

    if ("actions" not in st.session_state) or (not st.session_state["actions"]):
        set_actions(labels)

    col_table, col_margin, col_transformers = st.columns((4, 0.2, 1))
    cluster_table = render_cluster_table(df_clusters, col_table)
    selected_cluster_rows = cluster_table["selected_rows"]

    if selected_cluster_rows:
        st.session_state["selected_cluster_row"] = selected_cluster_rows[0]

    render_clusters_plot(df_mds, col_plot)
    render_cluster_chart(df_bm, col_cluster_chart)
    render_transformer_settings(col_transformers)

    btn_do_transformations = st.button("Aggregate and save")

    if btn_do_transformations:
        df_transformed = transform_bm(df_bm, cluster_table["data"])
        persist(df_transformed)
        render_transformed(df_transformed)
