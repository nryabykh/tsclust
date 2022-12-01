import pandas as pd
import streamlit as st
from pandas import DataFrame
from millify import millify

from common.static import upload_caption


def run():
    st.subheader("File upload")
    st.caption(upload_caption)

    file = st.file_uploader(
        label="Select file to upload",
        type='csv',
        label_visibility="collapsed",
        on_change=_reset_df
    )

    if not file:
        return

    if 'df' not in st.session_state:
        try:
            df = pd.read_csv(file)
            st.session_state['df'] = df
            st.success('File uploaded successfully. Please, check preview below and proceed to the dataset adjustment')
        except Exception as e:
            st.error('Failed to upload. Please check the full stack trace')
            with st.expander('Full exception message'):
                st.exception(e)
    _show_preview()


def _reset_df():
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'clicked' in st.session_state:
        del st.session_state['clicked']


def _show_preview():
    for _ in range(3):
        st.write("")

    df = st.session_state['df']
    dict_col_types = df.dtypes.apply(lambda x: x.name).to_dict()
    renamer = {k: f'{k}: {v}' for k, v in dict_col_types.items()}

    count = len(df.index)
    col_count = len(df.columns)
    mem_usage = df.memory_usage(deep=True).sum()

    st.markdown('#### Data preview')
    cols = st.columns(4)
    cols[0].metric(label='Number of entries', value=millify(count, precision=2))
    cols[1].metric(label='Number of columns', value=col_count)
    cols[2].metric(label='Memory usage', value=millify(mem_usage, prefixes=['kB', 'MB', 'GB']))

    st.caption('Dataset truncated to 100 lines')
    st.dataframe(df.sample(100).rename(renamer, axis=1))


st.set_page_config(layout='centered')
run()
