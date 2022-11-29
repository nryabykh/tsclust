from dataclasses import dataclass
from enum import Enum
from typing import List

import streamlit as st
from pandas import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS

from common.static import default_eps


@dataclass
class ClusteringResult:
    df: DataFrame
    labels: List[int]

    def set_actions(self) -> None:
        for label in self.labels:
            func, new_name = (
                (ClusterTransformAction.SUM, f"sum_cl_{label}")
                if label != -1
                else (ClusterTransformAction.KEEP, "")
            )
            st.session_state["actions"][label] = ClusterTransform(label, func, new_name)


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


class PearsonClustering:
    def __init__(self, df: DataFrame):
        self.df = df

    def get_correlations(self) -> DataFrame:
        return (
            self.df
            .fillna(self.df.mean())
            .corr()
            .fillna(0)
        )

    def get_distances(self) -> DataFrame:
        return 1 - self.get_correlations()

    @property
    def df_dist(self):
        return self.get_distances()

    def get_clusters(self, **kwargs) -> ClusteringResult:
        eps = kwargs.get('eps', default_eps)
        st.write(self.get_correlations())
        st.write(self.df_dist)
        dbscan = DBSCAN(eps=eps, min_samples=3, metric='precomputed').fit(self.df_dist)
        labels = dbscan.labels_
        df_clusters = (
            self.df_dist
            .assign(label=labels)
            .reset_index()
            .groupby("label")["variable"]
            .apply(list)
            .reset_index()
        )

        if "actions" in st.session_state:
            del st.session_state["actions"]

        st.session_state["actions"] = {}
        result = ClusteringResult(df=df_clusters, labels=labels)
        result.set_actions()
        return result

    def get_mds(self, labels=None) -> DataFrame:
        """
        Transforms each time series to the point on 2d-space applying multidimensional scaling.
        MDS keeps distances specified in distance matrix 'df_dist'.
        """
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1543)
        df_mds = DataFrame(mds.fit_transform(self.df_dist), columns=["x", "y"])
        df_mds["name"] = self.df_dist.index
        if labels is not None:
            df_mds['cluster'] = labels
        return df_mds
