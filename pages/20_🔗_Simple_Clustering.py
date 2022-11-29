import altair as alt
import streamlit as st
from pandas import DataFrame
from streamlit.delta_generator import DeltaGenerator

from common import styler, example
from common.static import dbscan_help, date_col
from pages.simple_clustering.algo import PearsonClustering
from pages.simple_clustering.buttons import set_btn_clicked
from pages.simple_clustering.options import DataOptions


def run(df):
    st.subheader("Simple time series clustering using Pearson correlation")
    # st.dataframe(df)
    options = _get_options(df)
    btn_start_clustering = st.button(
        "Make clusters",
        on_click=set_btn_clicked, kwargs={"btn_name": "btn_start_clustering"}
    )
    if not (btn_start_clustering or st.session_state.get("btn_start_clustering", False)):
        return

    cols = options.series
    df = df[df.index >= options.date.strftime('%Y-%m-%d')][cols]
    chart = alt.Chart(df.reset_index()).transform_fold(
        ['-node1-login', '-node2-login']
    ).mark_line().encode(
        x='dt:T',
        y='value:Q',
        color='key:N'
    )
    st.altair_chart(chart, use_container_width=True)
    pc = PearsonClustering(df)
    clusters = pc.get_clusters(eps=options.eps)
    df_mds = pc.get_mds(labels=clusters.labels)

    if ("actions" not in st.session_state) or (not st.session_state["actions"]):
        clusters.set_actions()

    col_plot, col_cluster_chart = st.columns(2)
    col_table, col_margin, col_transformers = st.columns((4, 0.2, 1))

    st.write(clusters.df, clusters.labels)
    cluster_table = render_cluster_table(clusters.df, col_table)
    # selected_cluster_rows = cluster_table["selected_rows"]
    #
    # if selected_cluster_rows:
    #     st.session_state["selected_cluster_row"] = selected_cluster_rows[0]
    #
    # render_clusters_plot(df_mds, col_plot)
    # render_cluster_chart(df, col_cluster_chart)
    # render_transformer_settings(col_transformers)
    #
    # btn_do_transformations = st.button("Aggregate and save")
    #
    # if btn_do_transformations:
    #     df_transformed = transform_bm(df_bm, cluster_table["data"])
    #     persist(df_transformed)
    #     render_transformed(df_transformed)


def _get_options(df: DataFrame) -> DataOptions:
    metrics_cols = list(df.columns)
    col_series, col_date, col_slider = st.columns((3, 1, 1))

    bm_selected = col_series.multiselect(
        label='Select series',
        options=metrics_cols,
        default=metrics_cols,
    )

    start_date = df.index.min()
    selected_start_date = col_date.date_input(
        label='Time series history started at',
        value=start_date,
    )

    selected_eps = col_slider.slider(
        label='Number of clusters adjustment',
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help=dbscan_help
    )

    return DataOptions(date=selected_start_date, eps=selected_eps, series=bm_selected)


def render_cluster_table(df_clusters: DataFrame, place: DeltaGenerator) -> dict:
    place.markdown("#### Clusters table")
    place.caption(("In this table you can find time series grouped by clusters. Click on the row to show charts "
                   "of all series in the cluster, as well as to specify aggregation settings."))

    df_clusters["agg_type"] = df_clusters["label"].apply(lambda x: st.session_state["actions"][x].func.value)
    df_clusters["agg_name"] = df_clusters["label"].apply(lambda x: st.session_state["actions"][x].new_name)
    df_clusters["variable"] = df_clusters["variable"].apply(lambda x: "; ".join(x))

    formatter = {
        "label": ("Cluster", {"pinned": "left", "width": "10"}),
        "agg_type": ("Action", {}),
        "agg_name": ("New name", {}),
        "variable": ("Series", {}),
    }

    with place:
        table = styler.draw_grid(
            df_clusters, formatter=formatter, selection="single", wrap_text=True
        )

    return table


st.set_page_config(layout='wide')
if 'df_clear' in st.session_state:
    run(st.session_state['df_clear'])
else:
    example.run()
    run(st.session_state['df_clear'])
    # st.info('Please, upload dataset')
