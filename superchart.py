"""
Helper for drawing multi-axis Altair charts within Streamlit application.
You can attach any number of lines to axis and combine multiple axis into a chart.
For each axis you can specify line options: captions, annotations, interpolation, captions' precision, ticks, etc...

Usage:

from superchart import Axis, SuperChart

df = pd.DataFrame()  # required cols: dt, variable, value. dt must be in datetime.

ax_q = Axis(
    name="Axis name",
    var_list=["pressure", "volume"],  # line names (values from 'variable' col)
    captions_interval=1,
    precision=1,
    time_format="%d/%m",
    tick_count="day"
)

SuperChart(df, [ax_q], caption_color="#ffffff").create(inplace=True)
"""


import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

import altair as alt
from altair import datum, Chart, Undefined
from pandas import DataFrame, Series
import streamlit as st

Axes = Dict[str, dict]

# Colors = d3.category10 + paired lighter colors from d3.category20
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]
DARK_GREY = '#44475a'


@dataclass
class Axis:
    name: str
    chart_type: str = None
    var_list: list = field(default_factory=list)
    captions_interval: Optional[int] = None  # 0 for auto interval, None for no captions
    show_annotations: bool = False
    annotations_col: str = "annotation"
    show_points: bool = True
    height: int = 120
    interpolation: str = "monotone"
    precision: int = -1  # digits after period. -1 if keep initial precision
    aggregation: str = ""
    time_format: str = "%Y-%m-%d"
    tick_count: Union[str, float] = Undefined

    def copy(self):
        return Axis(**self.__dict__)

    def set_name(self, newname: str):
        self.name = newname
        return self

    def set_vars(self, new_vars: List[str]):
        self.var_list = new_vars
        return self


class SuperChart:
    def __init__(self, df: DataFrame, axes: List[Axis], caption_color: str):
        self.VARIABLE_COL = "variable"
        self.VALUE_COL = "value"
        self.df = df
        self.axes = axes
        self.caption_color = caption_color
        self.domain = [c for c in itertools.chain.from_iterable(
            ax.var_list for ax in axes
        )]
        self.metric_colors = (
            list(zip(self.domain, COLORS))
            if len(self.domain) <= len(COLORS)
            else list(zip(self.domain, itertools.cycle(COLORS)))
        )
        self.zoom = alt.selection(type="interval", encodings=["x"])
        # self.auto_captions = self._get_auto_caption_pointers()

    def create(self, inplace: bool = False) -> Optional[List[Chart]]:
        charts = [self.create_single(ax) for ax in self.axes]
        if not inplace:
            return charts
        for chart in charts:
            st.write(chart)

    def create_single(self, ax: Axis):
        ax_name = ax.name

        # Prepare data
        var_filter = self.df["variable"].isin(ax.var_list)
        df_ax = self.df[var_filter].sort_values("dt").reset_index()
        df_ax = self._set_caption_pointers(df_ax, ax.captions_interval)
        df_ax["rounded_value"] = df_ax["value"] if ax.precision == -1 else df_ax["value"].round(ax.precision)
        df_ax = df_ax.rename({"variable": "Показатель"}, axis=1)

        max_value = df_ax["value"].max()

        ax_metric_colors = [(metric, color) for (metric, color) in self.metric_colors if metric in ax.var_list]
        ax_domain, ax_colors = zip(*ax_metric_colors)

        # Base chart. Colored lines
        color = alt.Color(
            'Показатель:N',
            scale=alt.Scale(domain=ax_domain, range=ax_colors)
        )
        base = alt.Chart(df_ax)

        lines = (
            base.mark_line(point=ax.show_points).encode(
                x=alt.X('dt:T', axis=alt.Axis(title=None, format=ax.time_format, tickCount=ax.tick_count)),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, max_value * 1.1])),
                color=color
            )
        )

        upper = lines.properties(height=ax.height, width=1200, title=ax_name)
        lower = lines.properties(height=30, width=1200)

        text = self._add_captions(lines)
        upper = upper + text

        if ax.show_annotations:
            if ax.annotations_col in df_ax.columns:
                df_ax["annotations"] = df_ax[ax.annotations_col]
                annotations = self._add_annotations(df_ax)
                upper = upper + annotations
            else:
                st.warning(f"No column '{ax.annotations_col}'")

        upper = (
            upper
            .encode(alt.X("dt:T", scale=alt.Scale(domain=self.zoom)))
            .properties(title=ax_name)
        )

        lower = self._add_colorized_zoom(lower)

        return upper & lower

    def _set_caption_pointers(self, df: DataFrame, caption_interval: Optional[int]) -> DataFrame:
        if caption_interval is None:
            df["show_captions"] = 0
            return df
        elif caption_interval == 0:
            return self._set_auto_caption_pointers(df)
        else:
            df["show_captions"] = df.index % caption_interval == 0
            return df

    def _set_auto_caption_pointers(self, df: DataFrame, caption_num: int = 10) -> DataFrame:
        steps = (
            df[self.VARIABLE_COL]
            .value_counts()
            .apply(lambda x: int((x - 0.1) // caption_num + 1))
            .rename_axis(self.VARIABLE_COL)
            .reset_index(name="steps")
        )
        df["row_number"] = df.groupby(self.VARIABLE_COL).cumcount()
        df_captions = df.merge(steps, on=self.VARIABLE_COL)
        df_captions["show_captions"] = df_captions["row_number"] % df_captions["steps"] == 0
        return df_captions

    def _add_captions(self, base: Chart) -> Chart:
        color = {"color": alt.value(self.caption_color)} if self.caption_color is not None else {}
        text_align = {"align": "center", "baseline": "top", "dx": 2, "dy": -15}
        return (
            base
            .mark_text(**text_align)
            .encode(text='rounded_value:Q', **color)
            .transform_filter(datum.show_captions == 1)
        )

    @staticmethod
    def _add_annotations(df) -> Chart:
        return (
            alt.Chart(df)
            .mark_rule(strokeDash=[12, 6], size=2)
            .encode(x="dt:T", color=alt.value("#0070ba"))
            .transform_filter(datum.annotations == 1)
        )

    def _add_colorized_zoom(self, lower_chart: Chart) -> Chart:
        lower_chart = lower_chart.encode(color=alt.condition(self.zoom, 'Показатель:N', alt.value(DARK_GREY)))
        lower_back = lower_chart.add_selection(self.zoom)
        lower_front = lower_chart.transform_filter(self.zoom).mark_line(color='Показатель:N')
        return lower_front + lower_back
