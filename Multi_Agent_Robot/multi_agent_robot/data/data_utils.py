import numpy as np
from Multi_Agent_Robot.multi_agent_robot.data.plotting import plot_3d_data, plot_2d_data
from Multi_Agent_Robot.multi_agent_robot.data.api import DataApi
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
        return agents_history


# todo
# add POMCP
# add sample quality to reward
# implement an oracle which intervenes as a function of cost - puts 1 or 0 in the belief
# maybe the play can address the oracle?
# train an NN to see if the env will need a intervention (Meta oracle)

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
        beliefs = hist['agent_beliefs']

        for i in range(0, len(beliefs[0]), 2):
            y_vecs.append(np.array(beliefs.apply(lambda belief: belief[i])))

    plot_2d_data(X, y_vecs, "baysian update beliefs", x_label="step", y_label="beliefs", y_labels=["rock1", "rock2", "rock3"])


if __name__ == '__main__':
    graph_baysian_agents_beliefs()
