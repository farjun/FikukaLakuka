from typing import List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_data(vecs: List[np.array], plot_name:str):
    x = vecs[:, 0]
    y = vecs[:, 1]
    z = vecs[:, 2]

    # Create a 3D figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vectors
    ax.scatter(x, y, z)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(plot_name)

    # Show the plot
    plt.show()

def plot_2d_data(X: np.array, y_vecs: List[np.array], plot_name:str, x_label:str, y_label:str, y_labels: List[str]):
    colors = ['red', 'blue', 'green', 'yellow', 'orange']
    for v, c, y_label in zip(y_vecs, colors, y_labels):
        plt.plot(X, v, color=c, label=y_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_name)

    plt.show()

if __name__ == '__main__':
    plot_3d_data()