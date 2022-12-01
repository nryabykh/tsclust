from functools import partial

from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype

from pages.adjust.adjustments import adjust_datetime, adjust_long, adjust_resample
import streamlit as st
import altair as alt


def run(df):
    adjust_funcs = []
    col_adj, col_results = st.columns((1, 2), gap='large')

    with col_adj:
        st.markdown('#### Adjustments')
        st.caption('Set all adjustments and click "Apply" button.')

        tab_date, tab_format, tab_resample = st.tabs(['Date column', 'Table format', 'Resampling'])

        with tab_date:
            date_col, date_format = _get_date_settings(df)
        adjust_funcs.append(partial(adjust_datetime, date_col, date_format))

        with tab_format:
            table_type = st.radio(
                label='Table format',
                options=('Wide', 'Long'),
                index=0 if len(df.columns) > 3 else 1,
                horizontal=True
            )
            if table_type == 'Long':
                variable_col, value_col = _get_table_settings(df, date_col)
                adjust_funcs.append(partial(adjust_long, variable_col, value_col, date_col))
            else:
                st.caption('Each column with numeric dtype will be processed as a separate time series')

        with tab_resample:
            resample_freq, resample_func = _get_resample_settings()
        adjust_funcs.append(partial(adjust_resample, resample_freq, resample_func))

        btn = st.button('Apply', on_click=_set_clicked)
        if btn:
            df_adj = _apply_adjustments(df, adjust_funcs)
            st.session_state['df_clear'] = df_adj

    with col_results:
        _show()

    if st.session_state.get('clicked', False):
        st.success('If resulting dataset seems well, please, proceed to the clustering')


def _get_date_settings(df) -> tuple[str, str]:
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
    return date_col, date_format


def _get_table_settings(df: DataFrame, date_col: str) -> tuple[str, str]:
    variable_col = st.multiselect(
        label='Column with variable names',
        options=sorted(set(c for c in df.columns if c != date_col))
    )
    value_col = st.selectbox(
        label='Column with time series values',
        options=[c for c in df.columns if c != date_col and is_numeric_dtype(df[c])]
    )
    return variable_col, value_col


def _get_resample_settings() -> tuple[str, str]:
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
    return resample_freq, resample_func


def _apply_adjustments(df: DataFrame, adjust_funcs: list[partial]) -> DataFrame:
    for func in adjust_funcs:
        try:
            df = func(df)
        except Exception as e:
            st.error(f'Cannot compete adjustment {func.func.__name__}. Skipped.')
            with st.expander('Full exception message'):
                st.exception(e)
    return df


def _show():
    st.markdown("#### Resulting dataset")

    tab_chart, tab_dataset, tab_init = st.tabs(['Result: Chart', 'Result: Table', 'Uploaded dataset'])

    with tab_init:
        if 'df' in st.session_state:
            df = st.session_state['df']
            if len(df.index) > 100:
                st.caption('Dataset truncated to 100 lines')
            st.dataframe(df.sample(min(100, len(df.index)), random_state=42), height=400, use_container_width=True)
        else:
            st.info('Please, return to the previous step and upload dataset')

    with tab_dataset:
        if 'df_clear' in st.session_state:
            df_clear = st.session_state['df_clear']
            if len(df_clear.index) > 100:
                st.caption('Resulting dataset truncated to 100 lines')
            st.dataframe(df_clear.sample(min(100, len(df.index)), random_state=42), height=400, use_container_width=True)
        else:
            st.info('Please, make adjustments and click "Apply" button, or explore an initial dataset on the right tab')

    with tab_chart:
        if 'df_clear' in st.session_state:
            df_clear = st.session_state['df_clear']
            if len(df_clear.index) > 100:
                st.caption('Chart based on the original result without any truncating')

            chart = alt.Chart(df_clear.reset_index()).transform_fold(
                [c for c in df_clear.columns if is_numeric_dtype(df_clear[c])]
            ).mark_line().encode(
                x='dt:T',
                y='value:Q',
                color='key:N'
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info('Please, make adjustments and click "Apply" button, or explore an initial dataset on the right tab')


def _set_clicked():
    st.session_state['clicked'] = True


st.set_page_config(layout='wide')
if 'df' in st.session_state:
    run(st.session_state['df'])
else:
    st.info('Please, upload dataset')
