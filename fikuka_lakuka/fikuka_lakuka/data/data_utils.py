from typing import List

import numpy as np

from fikuka_lakuka.fikuka_lakuka.data.plotting import plot_3d_data
from fikuka_lakuka.fikuka_lakuka.data.api import DataApi
from sklearn.manifold import LocallyLinearEmbedding

class DataUtils:

    def __init__(self, data_api: DataApi):
        self.data_api = data_api

    def cluster_states(self, agent: str, n_components=3):
        LLE = LocallyLinearEmbedding(n_components=n_components)
        state = self.data_api.get_all_states(agent, flatten_states=True)
        return LLE.fit_transform(state)

    def load_table_as_df(self, table_name):
        self.data_api


def run_state_clustering():
    data_api = DataApi()
    du = DataUtils(data_api)
    clustered_states = du.cluster_states("bbu")
    plot_3d_data(clustered_states, "baysian update states")

if __name__ == '__main__':
    run_state_clustering()
