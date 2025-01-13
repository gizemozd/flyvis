"""
    Script to create a connectome network class.
"""

from pathlib import Path
from typing import List, Dict
from numpy.typing import NDArray
import warnings
import numpy as np
import networkx as nx
import datamate
from datamate import Directory, Namespace, root

import flyvision
print("Data directory: ", flyvision.data_dir)

@root(flyvision.data_dir) # pointer to ./<data_dir>
class GroomingConnectome(Directory):
    """
    Connectome class for the MDN network.

    Attributes:
        nodes: Node properties
        edges: Edge properties
        input_indices: Indices of input neurons
        output_indices: Indices of output neurons
        unique_cell_types: Unique cell types in the connectome
    """

    class Config:
        adj_matrix_file: Path
        "Path of the adjacency matrix file"
        idx_name_file: Path
        "Path of the index name file"
        input_names: List[str] = ["JO-F", "JO-E", "JO-C"]
        "List of input neuron names"
        output_names: List[str] = [
            "ANTEN_MN1",
            "ANTEN_MN2",
            "ANTEN_MN3",
            "ANTEN_MN4",
            "ANTEN_MN5",
            "NECK_MN6",
            "NECK_MN7",
            "NECK_MN9",
            "NECK_MN10"
        ]
        "List of output neuron names"


    def __init__(self, config: Config) -> None:
        adj_matrix_file = config.adj_matrix_file
        idx_name_file = config.idx_name_file
        input_name = config.input_names
        output_name = config.output_names

        # adj matrix is post (rows) and pre (columns)
        adjacency_matrix = np.load(adj_matrix_file)
        # check if the adj matrix is symmetric
        half_ind = int(adjacency_matrix.shape[0] / 2)
        if not np.allclose(
            adjacency_matrix[:half_ind, :half_ind],
            adjacency_matrix[half_ind:, half_ind:], atol=1e-8
        ):
            warnings.warn("Ipsi matrix is not symmetric.")

        if not np.allclose(
            adjacency_matrix[half_ind:, :half_ind],
            adjacency_matrix[:half_ind, half_ind:], atol=1e-8
        ):
            warnings.warn("Contra matrix is not symmetric.")
        # check the orientation of the matrix
        assert sum(sum(adjacency_matrix[:, :10])) > 0, "Adj matrix orientation might be wrong."

        self.neuron_idx = np.load(idx_name_file).astype("S")

        graph = nx.from_numpy_array(
            # transpose
            adjacency_matrix.T,
            parallel_edges=False,
            create_using=nx.DiGraph,
        )

        self.adjacency_matrix = adjacency_matrix

        # Set up node indices and names
        print("Setting up the nodes")
        neurons_idx_types = self.get_node_types(graph)
        self.nodes.index = np.array(list(neurons_idx_types.keys()))
        self.nodes.type = np.array(list(neurons_idx_types.values())).astype("S")
        # U reading, S writing..
        self.unique_cell_types = np.unique(self.nodes.type[:].astype("U")).astype("S")

        # Set up the edges
        print("Setting up the edges")
        edge_dictionary = self.get_edge_properties(graph)
        self.edges.source_index = edge_dictionary["source"]
        self.edges.target_index = edge_dictionary["target"]
        self.edges.sign = edge_dictionary["sign"]
        self.edges.n_syn = edge_dictionary["syn_count"]
        number_of_edges = edge_dictionary["syn_count"].shape[0]

        # Automatically assigned from edge properties
        self.edges.source_type = self.nodes.type[:].astype("S")[self.edges.source_index[:]]
        self.edges.target_type = self.nodes.type[:].astype("S")[self.edges.target_index[:]]
        self.edges.edge_type = np.repeat("chem", number_of_edges).astype("S")
        print("Setting up the I/O")

        # Input output assignment
        self.input_indices = np.array(
            [
                idx
                for idx, nname in enumerate(np.array(self.neuron_idx).astype("U"))
                if any([iname in nname for iname in input_name])
            ]
        ).astype("int")
        self.output_indices = np.array(
            [
                idx
                for idx, nname in enumerate(np.array(self.neuron_idx).astype("U"))
                if any([iname in nname for iname in output_name])
            ]
        ).astype("int")

    def get_node_types(
        self,
        graph,
    ) -> Dict[int, str]:
        """Groups neurons into categories."""
        node_idx_type = {}

        for node_idx in graph.nodes():
            node_name = self.neuron_idx[node_idx].astype("U").replace(" ", "_")
            node_idx_type[node_idx] = node_name

        return node_idx_type

    def get_edge_properties(self, graph) -> Dict[str, NDArray]:
        """Get edge properties like source and target indices."""
        source_indices = []
        target_indices = []
        signs = []
        n_synapses = []

        for source_ind, target_ind in graph.edges():
            source_indices.append(source_ind)
            target_indices.append(target_ind)

            syn_count_signed = graph.get_edge_data(source_ind, target_ind)["weight"]

            syn_sign = np.sign(syn_count_signed)
            syn_count = np.abs(syn_count_signed)

            signs.append(syn_sign)
            n_synapses.append(syn_count)

            assert (
                syn_count_signed == np.array(self.adjacency_matrix)[target_ind, source_ind]
            ), f"{np.array(self.adjacency_matrix)[source_ind, target_ind]} is not equal to {syn_count} for {source_ind}, {target_ind}"

        return {
            "source": np.array(source_indices),
            "target": np.array(target_indices),
            "sign": np.array(signs),
            "syn_count": np.array(n_synapses),
        }

if __name__=="__main__":
    connectome_config=Namespace(
        type="GroomingConnectome",
        adj_matrix_file=Path("../data/training_dataset_2024_April/adj_matrix/symmetric_adj_matrix_max.npy"),
        idx_name_file=Path("../data/training_dataset_2024_April/adj_matrix/neuron_clusters.npy")
    )
    conn = GroomingConnectome(config=connectome_config)
