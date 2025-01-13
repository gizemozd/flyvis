import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from mycolorpy import colorlist as mcp

def plot_rate_sim(
    neuron_activity,
    duration,
    neuron_ind,
    time_step=1e-3,
    fig=None,
    ax=None,
    export_path=None,
    title="",
    xlims=None
):
    """Plot the neuron activity of a simulation. """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 15), dpi=100)

    v_min, v_max = neuron_activity[:, :-1].min(), neuron_activity[:, :-1].max()
    plt.imshow(neuron_activity[:, :-1], cmap="Blues", aspect="auto", vmin=v_min, vmax=v_max)
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")

    interval = 1e2

    plt.title("Network activity")

    plt.suptitle(title)

    plt.xticks(
        ticks=np.arange(0, duration + interval, interval),
        labels=np.arange(0, duration + interval, interval) * time_step
    )
    plt.yticks(
        ticks=range(len(neuron_ind)),
        labels=neuron_ind,
        rotation=0,
        fontsize=6,
    )

    if xlims is not None:
        plt.xlim(xlims)

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")


def plot_neurons(
    rate,
    duration,
    neuron_name,
    index_name,
    title='',
    export_path=None
):
    if not isinstance(index_name, list):
        index_name = list(index_name)

    neurons_of_interest = [
        index_name.index(neuron)
        for neuron in index_name
        if any([name.lower() in neuron.lower() for name in neuron_name])
    ]

    if 71 <= len(neurons_of_interest):
        nrows=9
    elif 51 <= len(neurons_of_interest) <= 70:
        nrows = 7
    elif 36 <= len(neurons_of_interest) < 51:
        nrows = 5
    elif 21 < len(neurons_of_interest) < 36:
        nrows = 4
    elif 13 <= len(neurons_of_interest) <= 21:
        nrows = 3
    elif 6 < len(neurons_of_interest) < 13:
        nrows = 2
    else:
        nrows = 1

    ncols = math.ceil(len(neurons_of_interest) / nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 1.5), sharey=True)
    axs = axs.flatten()

    for i, neuron_id in enumerate(neurons_of_interest):

        axs[i].plot(
            np.arange(duration - 1) * 1e-3,
            rate[neuron_id, :-1],
            label=index_name[neuron_id],
            color="black",
        )

        axs[i].legend()

    sns.despine()

    plt.suptitle(title)

    for i in range(1, ncols + 1):
        axs[-i].set_xlabel("Time (s)")

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")


def plot_jo_groups(
    rate,
    external_input,
    duration,
    jo_groups,
    index_name,
    sides=['R', 'L'],
    title='',
    export_path=None
):
    if not isinstance(index_name, list):
        index_name = list(index_name)

    fig, axs = plt.subplots(len(jo_groups), 2, figsize=(10, len(jo_groups) * 2))

    axs = axs.flatten()

    for i, jo_group in enumerate(jo_groups):
        jo_indices = [
            index_name.index(neuron)
            for neuron in index_name
            if jo_group.lower() in neuron.lower() and any(
                [side == neuron[-1] for side in sides]
            )
        ]

        colors = mcp.gen_color(cmap="binary", n=5 + len(jo_indices))

        for j, jo_ind in enumerate(jo_indices):
            axs[i * 2].plot(
                np.arange(duration - 1) * 1e-3,
                external_input[jo_ind, :-1],
                color=colors[j + 3],
                alpha=0.7,
            )
            axs[(i * 2) + 1].plot(
                np.arange(duration - 1) * 1e-3,
                rate[jo_ind, :-1],
                color=colors[j + 3],
                alpha=0.7,
            )
        axs[i * 2].set_ylabel(jo_group)
        axs[(i * 2) + 1].set_ylabel("Activity (AU)")

    axs[0].set_title("Given input")
    axs[1].set_title("Sensory neuron activities")

    axs[-1].set_xlabel("Time (s)")
    axs[-2].set_xlabel("Time (s)")
    plt.suptitle(title)

    plt.tight_layout()
    sns.despine()

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")
