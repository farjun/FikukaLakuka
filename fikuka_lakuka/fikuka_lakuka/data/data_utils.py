from typing import List

import numpy as np

from fikuka_lakuka.fikuka_lakuka.data.api import DataApi
from sklearn.manifold import LocallyLinearEmbedding

class DataUtils:

    def __init__(self, data_api: DataApi):
        self.data_api = data_api
        self.LLE = LocallyLinearEmbedding(n_components=3)

    def cluster_states(self, agent: str):
        state = self.data_api.get_all_states(agent, flatten_states=True)
        return self.LLE.fit_transform(state)
