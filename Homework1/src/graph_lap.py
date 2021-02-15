import numpy as np
from matplotlib import pyplot as plt

arr = np.ndarray


def comp_graph_lap(edgemat: arr) -> arr:
    dmat = np.zeros((edgemat.max() + 1, edgemat.max() + 1))
    adjmat = np.zeros(dmat.shape)
    for edge in edgemat:
        dmat[edge[0], edge[0]] += 1
        dmat[edge[1], edge[1]] += 1
        adjmat[edge[0], edge[1]] += 1
        adjmat[edge[1], edge[0]] += 1

    return np.subtract(dmat, adjmat)


if __name__ == "__main__":
    edgemat = np.genfromtxt("cell_graph.edgelist", dtype=int)
    graph_lap = comp_graph_lap(edgemat)
    print(graph_lap[graph_lap < 0])
    eigs = np.linalg.eigvals(graph_lap)
    print(eigs[eigs < 1])
    plt.hist(eigs, bins=50, edgecolor="black")
    plt.ylabel("Frequency")
    plt.xlabel("Eigenvalue")
    plt.title("Eigenvalues from Graph Laplacian")
    plt.savefig("eigenvalueshist.png")

