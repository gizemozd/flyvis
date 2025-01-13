import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from datamate import Namespace
from network_dynamics import GroomingDynamics, Stimulus
import flyvision

SEED = 20
flyvision.data_dir = f"../results/test_network_{SEED}"
print(flyvision.data_dir )

from grooming_connectome_network import GroomingConnectome

if __name__=="__main__":
    flyvision.network.Stimulus = Stimulus

    network = flyvision.network.Network(
        connectome=Namespace(
            type="GroomingConnectome",
            adj_matrix_file=Path("../data/training_dataset_2024_April/adj_matrix/symmetric_adj_matrix_max.npy"),
            idx_name_file=Path("../data/training_dataset_2024_April/adj_matrix/neuron_clusters.npy")
        ),
        dynamics=Namespace(
            type="GroomingDynamics",
            activation="relu",
        ),
        # parameters of nodes
        # put more parameters if you wanna train other things too
        node_config=Namespace(
            bias=Namespace(
                type="RestingPotential",
                groupby=["type"],
                initial_dist="Normal",
                mode="sample",
                clamp="non_negative",
                requires_grad=True,
                mean=0.04,  # 40 mV
                std=0.005,  # 5 mV std
                seed=SEED,
            ),
            # you need to change the things in initialization.py
            time_const=Namespace(
                type="TimeConstant",
                groupby=["type"],
                initial_dist="Value",
                value=0.03,  # 20 ms
                clamp="non_negative",
                requires_grad=True,
            ),
        ),
        # change the groupby here
        edge_config=Namespace(
            sign=Namespace(
                type="SynapseSign",
                initial_dist="Value",
                requires_grad=False,
                groupby=["source_type", "target_type"],
            ),
            syn_count=Namespace(
                type="SynapseCount",
                initial_dist="Lognormal",
                mode="mean",
                requires_grad=False,
                std=1.0,
                # Between each cluster, the synaptic count is averaged
                groupby=["source_type", "target_type"],
                seed=SEED,
            ),
            # definitely change how things are scaled
            syn_strength=Namespace(
                type="SynapseCountScaling",
                initial_dist="Value",
                requires_grad=True,
                scale_elec=0.1,
                scale_chem=0.1,
                clamp="non_negative",
                groupby=["source_type", "target_type", "edge_type"],  # add inhibitory, excitatory??
            ),
        ),
        connectome_class=GroomingConnectome,
        dynamics_class=GroomingDynamics,
    )

    neurons_index = np.load(Path("../data/training_dataset_2024_April/adj_matrix/neuron_clusters.npy"))
    neuron_to_silence = "WED"
    jof_indices = [i for i, neuron in enumerate(neurons_index) if neuron == neuron_to_silence]
    # Give some input to the network
    # x = torch.zeros(1, 1000, network.connectome.nodes.to_df().shape[0])
    # x[0, 100:200:10, :100] = 1
    # x[0, 400:600:5, 200:500:50] = 1
    # x[0, 700:900:2, 600:] = 1

    # # Run the network
    # voltage = network(x, dt=0.001)
    # voltage = voltage.detach().cpu().numpy().squeeze(0)

    # # Plot the results
    # for i in range(0,800,10):
    #     plt.plot(voltage[:, i], alpha=0.6, color='black')

    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.savefig("voltage_hardtanh.png")
    # plt.show()
    

    x = torch.zeros(1, 1000, network.connectome.nodes.to_df().shape[0])
    x[0, 400:600, jof_indices] = 1

    # Run the network
    voltage = network(x, dt=0.001)
    voltage_intact = voltage.detach().cpu().numpy().squeeze(0)

    # Silence JO-F 
    from biological_metrics import silence_neuron
    network_silenced = silence_neuron(network, neuron_to_silence, verbose=True)

    # from IPython import embed; embed()
    # Run the network
    voltage = network_silenced(x, dt=0.001)
    voltage_silenced = voltage.detach().cpu().numpy().squeeze(0)


    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(0, x.shape[2]):
        if i in jof_indices:
            axs[0,0].plot(voltage_intact[:, i], alpha=0.6, color='black')
            axs[0,1].plot(voltage_silenced[:, i], alpha=0.6, color='firebrick')
        else:
            axs[1,0].plot(voltage_intact[:, i], alpha=0.6, color='black')
            axs[1,1].plot(voltage_silenced[:, i], alpha=0.6, color='firebrick')

    axs[0,0].set_ylabel(f"{neuron_to_silence} neurons")
    axs[1,0].set_ylabel("Other neurons")
    
    axs[0,0].set_title("Intact")
    axs[0,1].set_title("Silenced network")


    plt.show()
