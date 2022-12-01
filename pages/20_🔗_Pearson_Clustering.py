import altair as alt
import streamlit as st
from pandas import DataFrame
from streamlit.delta_generator import DeltaGenerator

from common import example
from common.static import dbscan_eps_help, projection_caption, default_eps, default_min_samples, dbscan_min_samples_help
from pages.simple_clustering.algo import PearsonClustering
from pages.simple_clustering.buttons import set_btn_clicked
from pages.simple_clustering.options import DataOptions


def run(df):
    st.subheader("Simple time series clustering using Pearson correlation")
    options = _get_options(df)
    btn_start_clustering = st.button(
        "Make clusters",
        on_click=set_btn_clicked, kwargs={"btn_name": "btn_start_clustering"}
    )
    if not (btn_start_clustering or st.session_state.get("btn_start_clustering", False)):
        return

    cols = options.series
    df = df[df.index >= options.date.strftime('%Y-%m-%d')][cols]
    pc = PearsonClustering(df)
    clusters = pc.get_clusters(eps=options.eps, min_samples=options.min_samples)
    df_mds = pc.get_mds(labels=clusters.labels)

    if ("actions" not in st.session_state) or (not st.session_state["actions"]):
        clusters.set_actions()

    col_plot, _, col_cluster_chart = st.columns((1, 0.1, 1))
    col_table, col_margin, col_transformers = st.columns((4, 0.2, 1))

    render_cluster_table(clusters.df, col_table)
    #
    # if selected_cluster_rows:
    #     st.session_state["selected_cluster_row"] = selected_cluster_rows[0]
    #
    render_clusters_plot(df_mds, col_plot)
    render_cluster_chart(df, clusters.df, col_cluster_chart)
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
    col_series, _, col_date, _, col_slider = st.columns((3, 0.1, 0.8, 0.1, 1))

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
        label='Maximum distance between points in cluster',
        min_value=0.1,
        max_value=1.0,
        value=default_eps,
        step=0.05,
        help=dbscan_eps_help
    )

    selected_min_samples = col_slider.number_input(
        label='Minimum number of samples in cluster',
        min_value=1,
        value=default_min_samples,
        step=1,
        help=dbscan_min_samples_help
    )

    return DataOptions(date=selected_start_date, eps=selected_eps, min_samples=selected_min_samples, series=bm_selected)


def render_cluster_table(df_clusters: DataFrame, place: DeltaGenerator) -> dict:
    place.markdown("#### Clusters table")
    place.caption(("In this table you can find time series grouped by clusters. Click on the row to show charts "
                   "of all series in the cluster, as well as to specify aggregation settings."))

    df_clusters["agg_type"] = df_clusters["label"].apply(lambda x: st.session_state["actions"][x].func.value)
    df_clusters["agg_name"] = df_clusters["label"].apply(lambda x: st.session_state["actions"][x].new_name)

    place.dataframe(df_clusters, use_container_width=True)
    return {}


def render_clusters_plot(df_mds: DataFrame, place: DeltaGenerator) -> None:
    place.markdown("#### Time series projection to the plane")
    place.caption(projection_caption)

    chart = alt.Chart(df_mds).mark_circle().encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q", scale=alt.Scale(nice=True)),
        color="cluster:N",
        size=alt.value(100),
        tooltip=["name"]
    )
    text = alt.Chart(df_mds).mark_text(dx=5, dy=-10, color='#ffffff').encode(
        x=alt.X("x:Q"),
        y=alt.Y('y:Q'),
        detail='cluster:N',
        text='name',
        tooltip=alt.value(None)
    )
    place.altair_chart(chart, use_container_width=True)


def render_cluster_chart(df: DataFrame, df_clusters: DataFrame, place: DeltaGenerator) -> None:
    place.markdown("#### Time series of the selected cluster")
    place.caption("Here you can check the similarity of the time series from the same cluster visually.")
    labels_dict = df_clusters.set_index('label')['variable'].to_dict()
    labels = list(labels_dict.keys())
    cluster_label = place.radio(
        label='Select cluster to investigate',
        options=labels,
        index=0 if min(labels) == 0 or len(labels) == 1 else 1,
        horizontal=True
    )
    cluster_series = labels_dict.get(cluster_label, [])
    filtered_df = df[cluster_series]

    chart = (
        alt.Chart(filtered_df.reset_index())
        .mark_line()
        .transform_fold(cluster_series)
        .encode(
            x='dt:T',
            y='value:Q',
            color='key:N'
        )
    )
    place.altair_chart(chart, use_container_width=True)


st.set_page_config(layout='wide')

hide_table_row_index = """
    <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
"""

st.markdown(hide_table_row_index, unsafe_allow_html=True)

if 'df_clear' in st.session_state:
    run(st.session_state['df_clear'])
else:
    example.run()
    run(st.session_state['df_clear'])
    # st.info('Please, upload dataset')
