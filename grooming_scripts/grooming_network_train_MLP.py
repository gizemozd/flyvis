"""

    Code for training the connectome-constrained models

    Example usage:
    >>> python grooming_network_train_MLP.py \
        -dsp ../data/training_dataset_2024_April/training_data/all_flies_dataset_4_input_JOCEF_output_ant_neck.pkl \
        -ep ../results/connectome_original_relu_MLP_decoder

    Minimum adj matrix
    >>> python grooming_network_train_MLP.py \
        -dsp ../data/training_dataset_2024_April/training_data/all_flies_dataset_4_input_JOCEF_output_ant_neck.pkl \
        -ep ../results/connectome_min_sym_relu_MLP_decoder \
        -adj ../data/training_dataset_2024_April/adj_matrix/symmetric_adj_matrix_min.npy \
        -clust ../data/training_dataset_2024_April/adj_matrix/neuron_clusters.npy \
        --seed 0 --epochs 5000

    If maximum then symmetric_adj_matrix_max.npy
    If mean then symmetric_adj_matrix_mean.npy
    If original then original_adj_matrix.npy

"""
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from flyvision.dataset import GroomingDatasetClass, random_split
from flyvision.decoder_model import SimpleDecoder
from flyvision.loss_functions import combined_loss
from flyvision.visualization import plot_neurons
from flyvision.utils import nn_utils
from flyvision import model_utils
from datamate import Namespace

from network_dynamics import GroomingDynamics, Stimulus
import flyvision

# You can use tensorboard to monitor the training in real-time
# You can also try wandb, useful when running thingsd
# from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Run training",
    )
    parser.add_argument(
        "-dsp",
        "--training_dataset",
        type=str,
        default=None,
        help="Training dataset path",
    )
    parser.add_argument(
        "-adj",
        "--adj_matrix",
        type=str,
        default=None,
        help="Adjacency matrix path",
    )
    parser.add_argument(
        "-clust",
        "--cluster_names",
        type=str,
        default=None,
        help="Cluster names path",
    )
    parser.add_argument(
        "-ep",
        "--export_path",
        type=str,
        default=None,
        help="Export path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint",
    )
    return parser.parse_args()

ARGS = parse_args()
# Set the random seed
SEED = ARGS.seed
torch.manual_seed(SEED)

# Set the root directory
exp_path = ARGS.export_path
data_dir = Path(f'{exp_path}_seed_{SEED}')
flyvision.data_dir = data_dir

# Put this after setting the root dir
from grooming_connectome_network import GroomingConnectome

def main():
    """Main function."""
    index_name_list = np.load("../data/training_dataset_2024_April/adj_matrix/neuron_names_ordered.npy")

    adj_matrix_file = Path(ARGS.adj_matrix) if ARGS.adj_matrix else Path("../data/training_dataset_2024_April/adj_matrix/original_adj_matrix.npy")
    idx_name_file = Path(ARGS.cluster_names) if ARGS.cluster_names else Path("../data/training_dataset_2024_April/adj_matrix/neuron_clusters.npy")

    flyvision.network.Stimulus = Stimulus

    network = flyvision.network.Network(
        connectome=Namespace(
            type="GroomingConnectome",
            adj_matrix_file=adj_matrix_file,
            idx_name_file=idx_name_file,
            input_names=["JO-F", "JO-E", "JO-C"],
            output_names=[
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
        ),
        dynamics=Namespace(
            type="GroomingDynamics",
            activation="relu"
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
                # penalize=Namespace(activity=True),
                seed=SEED,
            ),
            # you need to change the things in initialization.py
            time_const=Namespace(
                type="TimeConstant",
                groupby=["type"],
                initial_dist="Value",
                value=0.03,  # 30 ms
                clamp=(0.0, 0.15), # 0-150 ms
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

    # Network check
    number_of_params = nn_utils.n_params(network)

    with open(ARGS.training_dataset, "rb") as f:
        dataset_dict = pickle.load(f)

    input_dataset = dataset_dict["input"]
    output_dataset = dataset_dict["output"]
    mask_dataset = dataset_dict["mask"]

    entire_dataset = GroomingDatasetClass(input_dataset, output_dataset, mask_dataset)

    # Here we set the seeds to 0 to train all models on the same dataset
    train_ds, test_val_ds = random_split(entire_dataset, 0.7, random_seed=1)
    valid_ds, test_ds = random_split(test_val_ds, 0.7, random_seed=1)

    input_ind = np.array(network.connectome.input_indices).astype("int")

    output_ind = np.array(network.connectome.output_indices).astype("int")
    output_ind_left = output_ind[:output_ind.shape[0] // 2]
    output_ind_right = output_ind[output_ind.shape[0] // 2:]

    assert output_ind_right.shape == output_ind_left.shape, "Output indices do not have the same shape."

    output_ant_right = [
        ind for ind in output_ind_right
        if 'ANTEN' in np.array(network.connectome.neuron_idx).astype('U')[ind]
    ]
    output_ant_left = [
        ind for ind in output_ind_left
        if 'ANTEN' in np.array(network.connectome.neuron_idx).astype('U')[ind]
    ]
    output_neck = [
        ind for ind in output_ind
        if 'NECK' in np.array(network.connectome.neuron_idx).astype('U')[ind]
    ]

    assert len(output_ant_left) == len(output_ant_right) == 5, "Right and left anten. output indices are not equal"
    assert len(output_neck) == 8, "There should be 8 neck motor neurons in total"

    # hidden size: store n info - two layers
    decoder_antenna = SimpleDecoder(
        activation_function="hardtanh",
        in_features=len(output_ant_left),
        hidden_features=len(output_ant_left) * 2,
        out_features=1,
        dropout=0.2
    )
    # decoder neck
    decoder_neck = SimpleDecoder(
        activation_function="hardtanh",
        in_features=len(output_neck),
        hidden_features=len(output_neck) * 2,
        out_features=1,
        dropout=0.25
    )

    optimizer = torch.optim.Adam(
        [*network.parameters(),
         *decoder_antenna.parameters(),
         *decoder_neck.parameters()],
        lr=1e-4, amsgrad=True
    )

    dataloader_train = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator(device=flyvision.device),
        num_workers=0,
    )

    dataloader_valid = DataLoader(
        valid_ds,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator(device=flyvision.device),
        num_workers=0,
    )

    dataloader_test = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator(device=flyvision.device),
        num_workers=0,
    )

    loss_func = combined_loss

    if ARGS.checkpoint:
        print(
            f"Loading the model from checkpoint {ARGS.checkpoint}..."
        )
        network = model_utils.load_model(data_dir, network, f"checkpoint{ARGS.checkpoint}_network_")
        decoder_antenna = model_utils.load_model(data_dir, decoder_antenna, f"checkpoint{ARGS.checkpoint}_decoder_antenna_")
        decoder_neck = model_utils.load_model(data_dir, decoder_neck, f"checkpoint{ARGS.checkpoint}_decoder_neck_")
        start_epoch = ARGS.checkpoint + 1
    else:
        start_epoch = 0

    N_EPOCHS = ARGS.epochs
    TIME_STEP = 0.001


    ############################## TRAINING ##############################
    if not data_dir.is_dir() or not (data_dir / "checkpoint5000_network_model_state_dict.pkl").is_file():
        if not data_dir.is_dir():
            data_dir.mkdir(exist_ok=True, parents=True)

        train_loss_list = []
        valid_loss_list = []

        time0 = time.time()

        print("Training the model...")
        print('-'*40)
        print("Number of fixed parameters:", number_of_params.fixed)
        print("Number of free parameters:", number_of_params.free)
        print('-'*40)

        min_valid_loss = np.inf

        for epoch in range(start_epoch, N_EPOCHS + 1):
            train_loss_epoch = 0
            valid_loss_epoch = 0

            # Training
            network.train()
            decoder_antenna.train()
            decoder_neck.train()
            for data in dataloader_train:
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # Input, output, mask
                x, y, mask = data
                # Separate output into right and left
                y_ant_right = y[..., 0].unsqueeze(-1)
                y_ant_left = y[..., 1].unsqueeze(-1)
                y_neck = y[..., 2].unsqueeze(-1)
                # Separate network response based on the output
                voltage = network(x, dt=TIME_STEP)
                voltage_output_neurons_right = voltage[..., output_ant_right]
                voltage_output_neurons_left = voltage[..., output_ant_left]
                voltage_output_neurons_neck = voltage[..., output_neck]
                # Make predictions
                y_pred_ant_right = decoder_antenna(voltage_output_neurons_right)
                y_pred_ant_left = decoder_antenna(voltage_output_neurons_left)
                y_pred_neck = decoder_neck(voltage_output_neurons_neck)
                # Calculate the loss and its gradients
                loss = loss_func(
                    y_ant_right, y_pred_ant_right, mask=mask
                ) + \
                    loss_func(
                        y_ant_left,y_pred_ant_left, mask=mask
                ) + \
                    loss_func(
                        y_neck, y_pred_neck, mask=mask
                )
                loss.backward()
                train_loss_epoch += loss.item()
                optimizer.step()

            # Set all the models at the evaluation mode
            network.eval()
            decoder_antenna.eval()
            decoder_neck.eval()
            # Validation
            with torch.no_grad():
                for data in dataloader_valid:
                    # Input, output, mask
                    x, y, mask = data
                    # Separate output into right and left
                    y_ant_right = y[..., 0].unsqueeze(-1)
                    y_ant_left = y[..., 1].unsqueeze(-1)
                    y_neck = y[..., 2].unsqueeze(-1)
                    # Separate network response based on the output
                    voltage = network(x, dt=TIME_STEP)
                    voltage_output_neurons_right = voltage[..., output_ant_right]
                    voltage_output_neurons_left = voltage[..., output_ant_left]
                    voltage_output_neurons_neck = voltage[..., output_neck]
                    # Make predictions
                    y_pred_ant_right = decoder_antenna(voltage_output_neurons_right)
                    y_pred_ant_left = decoder_antenna(voltage_output_neurons_left)
                    y_pred_neck = decoder_neck(voltage_output_neurons_neck)
                    # Calculate the loss and its gradients
                    loss = loss_func(
                        y_ant_right, y_pred_ant_right, mask=mask
                    ) + \
                        loss_func(
                            y_ant_left,y_pred_ant_left, mask=mask
                    ) + \
                        loss_func(
                            y_neck, y_pred_neck, mask=mask
                    )

                    valid_loss_epoch += loss.item()

            train_loss_epoch_avg = train_loss_epoch / len(dataloader_train)
            valid_loss_epoch_avg = valid_loss_epoch / len(dataloader_valid)

            # writer.add_scalars(
            #     'loss', {
            #         'training': train_loss_epoch_avg,
            #         'validation': valid_loss_epoch_avg,
            #     }, epoch
            # )

            print("Epoch {:d} / {:d}: time={:.3f}h, training loss={:.5f}, validation loss={:.5f}".format(
                    epoch + 1,
                    N_EPOCHS,
                    (time.time() - time0) / 3600,
                    train_loss_epoch_avg,
                    valid_loss_epoch_avg,
                ), end="\r",
            )

            if valid_loss_epoch < min_valid_loss and epoch > 500:
                print("\nValidation loss decreased at epoch {:d}: {:.3f} --> {:.3f}\nSaving the model...".format(
                        epoch + 1,
                        min_valid_loss,
                        valid_loss_epoch
                    )
                )
                min_valid_loss = valid_loss_epoch
                model_utils.save_model(data_dir, network, f"best_model_network_")
                model_utils.save_model(data_dir, decoder_antenna, f"best_model_decoder_antenna_")
                model_utils.save_model(data_dir, decoder_neck, f"best_model_decoder_neck_")

            if epoch % 100 == 0:
                train_loss_list.append(train_loss_epoch_avg)
                valid_loss_list.append(valid_loss_epoch_avg)

            if epoch % 250 == 0:
                model_utils.save_model(data_dir, network, f"checkpoint{epoch}_network_")
                model_utils.save_model(data_dir, decoder_antenna, f"checkpoint{epoch}_decoder_antenna_")
                model_utils.save_model(data_dir, decoder_neck, f"checkpoint{epoch}_decoder_neck_")
                # torch.cuda.empty_cache()

        print("\nDone. Training took {:.3f} h.".format((time.time() - time0) / 3600))
        # writer.close()

        # Save data
        model_utils.save_model(data_dir, network, "network_")
        model_utils.save_model(data_dir, decoder_antenna, "decoder_antenna_")
        model_utils.save_model(data_dir, decoder_neck, "decoder_neck_")
        # Save loss
        np.save(data_dir / "loss_train.npy", train_loss_list)
        np.save(data_dir / "loss_valid.npy", valid_loss_list)

        fig, ax = plt.subplots()
        plt.plot(range(len(train_loss_list)), train_loss_list, color="navy", label="training loss")
        plt.plot(range(len(valid_loss_list)), valid_loss_list, color="firebrick", label="validation loss")
        plt.xticks(
            [ep for ep in range(len(train_loss_list))],
            labels=[ep*100 for ep in range(len(train_loss_list))]
        )
        plt.ylabel("Loss (A.U.)")
        plt.xlabel("Epochs (A.U.)")
        plt.title("Training and validation losses over epochs")
        fig.savefig(data_dir / "loss.png", bbox_inches="tight")
        plt.close()

    else:
        print("Model is already trained, loading the values...")
        network = model_utils.load_model(data_dir, network, "network_")
        decoder_antenna = model_utils.load_model(data_dir, decoder_antenna, "decoder_antenna_")
        decoder_neck = model_utils.load_model(data_dir, decoder_neck, "decoder_neck_")

    # Set all the models at the evaluation mode
    network.eval()
    decoder_antenna.eval()
    decoder_neck.eval()

    with torch.no_grad():
        running_loss = 0
        for i, data in enumerate(dataloader_test):
            x, y, mask = data

            voltage = network(x, dt=TIME_STEP)
            # Separate the output based on the body part
            y_ant_right = y[..., 0].unsqueeze(-1)
            y_ant_left = y[..., 1].unsqueeze(-1)
            y_neck = y[..., 2].unsqueeze(-1)
            # Separate the voltage
            voltage_output_neurons_right = voltage[..., output_ant_right]
            voltage_output_neurons_left = voltage[..., output_ant_left]
            voltage_output_neurons_neck = voltage[..., output_neck]
            # Make predictions
            y_pred_ant_right = decoder_antenna(voltage_output_neurons_right)
            y_pred_ant_left = decoder_antenna(voltage_output_neurons_left)
            y_pred_neck = decoder_neck(voltage_output_neurons_neck)

            # Calculate the loss and its gradients
            loss = loss_func(
                y_ant_right, y_pred_ant_right, mask=mask
            ) + \
                loss_func(
                    y_ant_left,y_pred_ant_left, mask=mask
            ) + \
                loss_func(
                    y_neck, y_pred_neck, mask=mask
            )

            running_loss += loss.item()

            print("Test loss:", loss.item())

            fig, axs = plt.subplots(1,3, figsize=(7,2))
            axs[0].plot(y_pred_ant_right[0, :, 0].cpu().detach().numpy(), color="#E16E00")
            axs[1].plot(y_pred_ant_left[0, :, 0].cpu().detach().numpy(), color="#00008B")
            axs[2].plot(y_pred_neck[0, :, 0].cpu().detach().numpy(), label="pred.", color="black")
            axs[0].plot(
                y_ant_right[0, :, 0].cpu().detach().numpy(),
                color="#FFA500",
            )
            axs[1].plot(
                y_ant_left[0, :, 0].cpu().detach().numpy(),
                color="#ADD8E6",
            )
            axs[2].plot(
                y_neck[0, :, 0].cpu().detach().numpy(),
                label="ground truth",
                color="grey",
            )
            for ax_no, title in enumerate(['Ant. right', 'Ant. left', 'Neck']):
                axs[ax_no].set_xlabel('Time (ms)')
                axs[ax_no].set_ylabel('Activity (A.U.)')
                axs[ax_no].set_title(title)

            fig.savefig(data_dir / f"test_results_{i}.png", bbox_inches="tight")
            plt.close()

            # voltage_network = nn.ReLU()(voltage)
            v_rest = voltage.cpu().detach().numpy()
            duration = v_rest.shape[1]

            plot_neurons(
                v_rest.T,
                duration,
                neuron_name=["ANTEN_MN", "NECK_MN"],
                title="Motor neuron activies",
                index_name=index_name_list,
                export_path=data_dir / f"mn_rates_{i}.png",
            )
            plt.close()

            plot_neurons(
                v_rest.T,
                duration,
                neuron_name=["ADN", "ABN"],
                title="aDN and aBN activies",
                index_name=index_name_list,
                export_path=data_dir / f"adn_abn_rates_{i}.png",
            )
            plt.close()

        print("Avg test loss: ", running_loss / len(dataloader_test))


if __name__ == "__main__":
    main()
