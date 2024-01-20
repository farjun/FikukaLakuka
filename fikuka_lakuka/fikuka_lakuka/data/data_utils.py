from typing import List

import numpy as np

from fikuka_lakuka.fikuka_lakuka.data.plotting import plot_3d_data, plot_2d_data
from fikuka_lakuka.fikuka_lakuka.data.api import DataApi
from sklearn.manifold import LocallyLinearEmbedding

class DataUtils:

    def __init__(self, data_api: DataApi):
        self.data_api = data_api

    def cluster_states(self, agent: str, n_components=3):
        LLE = LocallyLinearEmbedding(n_components=n_components)
        state = self.data_api.get_all_states(agent, flatten_states=True)
        return LLE.fit_transform(state)

    def get_history_by_agent(self):
        agents_history = self.data_api.get_history(as_df=True)
        for agent_hist in agents_history:
            agent_hist
        return agents_history

    def load_table_as_df(self, table_name):
        self.data_api

# todo
# belief states, rewards, check converge to truth value
# dijstra between good rocks vs real route is the optimal efficiency distance
# implement an oracle which intervenes as a function of cost

# ofra and sarit kraus




def run_state_clustering():
    data_api = DataApi(db_name="run_1")
    du = DataUtils(data_api)
    clustered_states = du.cluster_states("bbu")
    plot_3d_data(clustered_states, "baysian update states")

def graph_baysian_agents_beliefs(db_name="run_1"):
    data_api = DataApi(db_name=db_name)
    du = DataUtils(data_api)
    agents_histories = du.get_history_by_agent()
    y_vecs = list()
    X = agents_histories[0]['step']

    for hist in agents_histories:
        if hist['cur_agent'][0] not in [1]:
            continue

        beliefs = hist['agent_beliefs']

        for i in range(0, len(beliefs[0]), 2):
            y_vecs.append(np.array(beliefs.apply(lambda belief: belief[i])))

    plot_2d_data(X, y_vecs, "baysian update beliefs", x_label="step", y_label="beliefs", y_labels=["rock1", "rock2", "rock3"])


if __name__ == '__main__':
    graph_baysian_agents_beliefs()
