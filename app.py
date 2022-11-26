import streamlit as st

from static import caption


def app():
    st.header("Business metrics preprocessing")

    set_session_state()
    st.caption(caption)

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