import math

import altair as alt
import streamlit as st
from pandas import DataFrame
from streamlit.delta_generator import DeltaGenerator

from common import example, styler
from common.static import dbscan_eps_help, projection_caption, default_eps, default_min_samples, dbscan_min_samples_help
from pages.simple_clustering.algo import PearsonClustering, ClusterTransformAction, ClusterTransform
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

    cols = [c for c in df.columns if c not in options.excluded_series]
    df = df[df.index >= options.date.strftime('%Y-%m-%d')][cols]
    pc = PearsonClustering(df)
    clusters = pc.get_clusters(eps=options.eps, min_samples=options.min_samples)
    df_mds = pc.get_mds(labels=clusters.labels)

    if ("actions" not in st.session_state) or (not st.session_state["actions"]):
        clusters.set_actions()

    col_plot, col_cluster_chart = st.columns((1, 1), gap='large')
    col_table, col_transformers = st.columns((1, 1), gap='large')

    render_cluster_table(clusters.df, col_table)
    render_clusters_plot(df_mds, col_plot)
    render_cluster_chart(df, clusters.df, col_cluster_chart)
    render_transformer_settings(clusters.df, col_transformers)

    btn_do_transformations = st.button("Aggregate")

    if btn_do_transformations:
        df_transformed = transform_ts(df, clusters.df)
        render_transformed(df_transformed)
        st.download_button(
            label='Download the result (CSV)',
            data=_get_csv(df_transformed),
            file_name='aggregated_time_series.csv',
            mime='text/csv'
        )


@st.experimental_memo(ttl=3600)
def _get_csv(df: DataFrame):
    return df.to_csv().encode('utf-8')


def _get_options(df: DataFrame) -> DataOptions:
    metrics_cols = list(df.columns)
    col_series, col_date, col_slider, col_samples = st.columns(4, gap='medium')

    excluded = col_series.multiselect(
        label='Exclude series',
        options=metrics_cols,
        default=None,
    )

    selected_start_date = col_date.date_input(
        label='Time series history started at',
        value=df.index.min(),
    )

    selected_eps = col_slider.slider(
        label='Maximum distance between points in cluster',
        min_value=0.1,
        max_value=1.0,
        value=default_eps,
        step=0.05,
        help=dbscan_eps_help
    )

    selected_min_samples = col_samples.number_input(
        label='Minimum number of samples in cluster',
        min_value=1,
        value=default_min_samples,
        step=1,
        help=dbscan_min_samples_help
    )

    return DataOptions(date=selected_start_date, eps=selected_eps, min_samples=selected_min_samples,
                       excluded_series=excluded)


def render_cluster_table(df_clusters: DataFrame, place: DeltaGenerator):
    place.markdown("#### Clusters table")
    place.caption(("In this table you can find time series grouped by clusters. Click on the row to show charts "
                   "of all series in the cluster, as well as to specify aggregation settings."))

    # place.table(df_clusters.set_index('label'))
    labels_dict = df_clusters.set_index('label')['variable'].to_dict()
    row_height = max(math.ceil(len(', '.join(v)) / 100) for k, v in labels_dict.items())
    with place:
        styler.draw_grid(
            df=df_clusters,
            formatter={'label': ('Label', {'width': 30}), 'variable': ('List of time series', {})},
            selection=None, fit_columns=True, wrap_text=True, row_height=row_height * 30)


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


def render_transformer_settings(df_clusters: DataFrame, col_transformers: DeltaGenerator):
    col_transformers.markdown("#### Transformations")
    col_transformers.caption("""Aggregate time series from the same cluster into the single new one. Cluster -1 
    stands for outlier. It would make sense to consider options 'Keep all originals' or 'Drop all' for it.""")

    labels_dict = df_clusters.set_index('label')['variable'].to_dict()
    labels = list(labels_dict.keys())
    actions = st.session_state['actions']
    is_viewed = [k.split('_')[-1] for k, v in st.session_state.items() if k.startswith('viewed_') and v]
    tabs = col_transformers.tabs([f'{"✅" if str(c) in is_viewed else ""} Cluster #{c}' for c in labels])
    for tab, label in zip(tabs, labels):
        with tab:
            funcs = [ct for ct in ClusterTransformAction]
            default_ix = funcs.index(actions[label].func.value)
            default_new_name = actions[label].new_name

            func = st.radio(
                label="Action",
                options=funcs,
                index=default_ix,
                format_func=lambda x: x.value,
                key=f"selected_action_{label}",
                horizontal=True
            )
            if func in [ClusterTransformAction.SUM, ClusterTransformAction.MEAN]:
                new_name = st.text_input(
                    label="New metric name",
                    value=default_new_name,
                    key=f"selected_new_name_{label}")
            elif func == ClusterTransformAction.ONE:
                new_name = st.selectbox(
                    label="Select a metric",
                    options=labels_dict.get(label),
                    key=f"selected_metric_{label}")
            else:
                new_name = ""
                st.caption('No need a new name')
            viewed = st.checkbox('Viewed', key=f'viewed_{label}')
            transform = ClusterTransform(label, func, new_name, viewed)
            st.session_state['actions'][label] = transform


def transform_ts(df: DataFrame, df_clusters: DataFrame) -> DataFrame:
    actions = st.session_state['actions']
    for cluster in df_clusters.to_dict(orient='records'):
        label, series = cluster['label'], cluster['variable']
        transform = actions[label]

        if transform.func == ClusterTransformAction.SUM:
            df[transform.new_name] = df[series].sum(axis=1)
            df = df.drop(series, axis=1)
        elif transform.func == ClusterTransformAction.MEAN:
            df[transform.new_name] = df[series].mean(axis=1)
            df = df.drop(series, axis=1)
        elif transform.func == ClusterTransformAction.ONE:
            to_drop = [c for c in series if c != transform.new_name]
            df = df.drop(to_drop, axis=1)
        elif transform.func == ClusterTransformAction.KEEP:
            pass
        elif transform.func == ClusterTransformAction.DROP:
            df = df.drop(series, axis=1)
        else:
            pass

    return df


def render_transformed(df: DataFrame, col_num: int = 3) -> None:
    st.markdown("#### Aggregation result")
    chart = alt.Chart(df.reset_index()).mark_line().transform_fold(list(df.columns)).encode(
        x=alt.X("dt:T", title=None),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("key:N")
    ).properties(
        width=round(1000 / col_num),
        height=130
    ).facet(
        "key:N",
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
    st.altair_chart(chart, use_container_width=True)


st.set_page_config(layout='wide')

if 'df_clear' in st.session_state:
    run(st.session_state['df_clear'])
else:
    example.run()
    run(st.session_state['df_clear'])
    # st.info('Please, upload dataset')
