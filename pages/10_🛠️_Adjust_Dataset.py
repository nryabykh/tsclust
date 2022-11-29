from functools import partial
from pandas.core.dtypes.common import is_numeric_dtype

from common import styler
from pages.adjust.adjustments import adjust_datetime, adjust_long, adjust_resample
import streamlit as st
import altair as alt


def run(df):
    _show_preview(df)
    adjust_funcs = []

    st.markdown('#### Adjustments')
    st.caption('Set all adjustments and click "Apply" button.')
    col_date, _, col_format, _, col_resample = st.columns((1, 0.3, 1, 0.3, 1))

    with col_date:
        date_col = st.selectbox(
            label='Date column',
            options=df.columns
        )
        date_format = st.text_input(
            label="Date format",
            value="s" if df[date_col].dtype == "int64" else "%Y-%m-%d",
            help="""D|s|ms|us|ns - if parse date in unix time (int64 col), date format such as '%Y-%m-%s' - if parse 
            string """
        )
    adjust_funcs.append(partial(adjust_datetime, date_col, date_format))

    with col_format:
        table_type = st.radio(
            label='Table format',
            options=('Wide', 'Long'),
            index=0 if len(df.columns) > 3 else 1,
            horizontal=True
        )
        if table_type == 'Long':
            variable_col = st.multiselect(
                label='Column with variable names',
                options=sorted(set(c for c in df.columns if c != date_col))
            )
            value_col = st.selectbox(
                label='Column with time series values',
                options=[c for c in df.columns if c != date_col and is_numeric_dtype(df[c])]
            )
            adjust_funcs.append(partial(adjust_long, variable_col, value_col, date_col))
        else:
            st.caption('Each column with numeric dtype will be processed as a separate time series')

    with col_resample:
        resample_freq = st.text_input(
            label='Resample frequency',
            value='1D',
            help="Format: number+modifier (s, min, H, D, W, M, full list you can find there: "
                 "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects_ "
        )
        resample_func = st.selectbox(
            label='Resample function',
            options=('sum', 'min', 'mean', 'max', 'median', 'first', 'last')
        )
    adjust_funcs.append(partial(adjust_resample, resample_freq, resample_func))

    btn = st.button('Apply', on_click=_set_clicked)
    if btn:
        for func in adjust_funcs:
            try:
                df = func(df)
            except Exception as e:
                st.error(f'Cannot compete adjustment {func.func.__name__}. Skipped.')
                with st.expander('Full exception message'):
                    st.exception(e)
        st.session_state['df_clear'] = df

    if st.session_state.get('clicked', False):
        _show(st.session_state['df_clear'])


def _show_preview(df):
    # st.markdown('#### Uploaded dataset')
    col_preview, col_rows, _ = st.columns((1, 1, 3))
    show_preview = col_preview.checkbox('Show uploaded dataset', value=False)
    show_all = col_rows.checkbox('Show all rows', value=False, disabled=not show_preview)
    if show_preview:
        st.dataframe(df.head(5) if not show_all else df, use_container_width=True)


def _show(df):
    col_dataset, col_chart, _ = st.columns((1, 1, 3))
    show_dataset = col_dataset.checkbox('Show dataset', value=False)
    show_chart = col_chart.checkbox('Show chart', value=True)

    if show_dataset:
        st.dataframe(df, use_container_width=True)

    if show_chart:
        chart = alt.Chart(df.reset_index()).transform_fold(
            [c for c in df.columns if is_numeric_dtype(df[c])]
        ).mark_line().encode(
            x='dt:T',
            y='value:Q',
            color='key:N'
        )
        st.altair_chart(chart, use_container_width=True)


def _set_clicked():
    st.session_state['clicked'] = True


st.set_page_config(layout='wide')
if 'df' in st.session_state:
    run(st.session_state['df'])
else:
    st.info('Please, upload dataset')
