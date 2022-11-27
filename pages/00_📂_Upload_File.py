import pandas as pd
import streamlit as st

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
            st.success('File uploaded successfully. Please, proceed to the dataset adjustment')
        except Exception as e:
            st.error('Failed to upload. Please check the full stack trace')
            with st.expander('Full exception message'):
                st.exception(e)


def _reset_df():
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'clicked' in st.session_state:
        del st.session_state['clicked']


run()
