import torch
from torch import nn

from datamate import Namespace

from flyvision.dynamics import NetworkDynamics


ACT_FUNC = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "leakyrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "hardtanh": nn.Hardtanh(min_val=0.0, max_val=10.0),
}


class Stimulus:
    def __init__(*args, **kwargs):
        pass

class GroomingDynamics(NetworkDynamics):
    """Passive point neurons with instantaneous graded release synapses."""

    class Config:
        activation: str = "relu"

    def __init__(self, config: Config):
        self.activation = ACT_FUNC[config.activation]

    def write_derived_params(self, params, **kwargs):
        """Weights are the product of the sign, synapse count, and strength."""
        params.edges.weight = params.edges.sign * params.edges.syn_count * params.edges.syn_strength

    def write_initial_state(self, state, params, **kwargs):
        """Initial state is the bias."""
        state.nodes.activity = params.nodes.bias

    def write_state_velocity(self, vel, state, params, target_sum, x_t, dt, **kwargs):
        """
            Velocity is the bias plus the sum of the weighted rectified inputs.
            state.nodes.activity -> V(t)
            vel.nodes.activity -> dV(t)
        """
        # We add noise of variance 0.02
        noise = (
            torch.randn(
                *x_t.size(),
                dtype=torch.float32,
            )
            * 0.01
        )

        # V(t) with Euler integration
        vel.nodes.activity = (
            1
            / torch.max(params.nodes.time_const, torch.tensor(dt).float())
            * (
                - state.nodes.activity  # V(t)
                - params.nodes.bias  #  Resting potential
                + target_sum(
                    params.edges.weight * self.activation(state.sources.activity)
                )  #  W * r(t) - incoming current
                + x_t  # input u(t)
                + noise # noise
            )
        )

    def currents(self, state, params):
        """Return the internal chemical current."""
        return params.edges.weight * self.activation(state.sources.activity)

if __name__=="__main__":
    config = Namespace(
        type="GroomingDynamics",
        activation="relu"
    )
    dynamics = GroomingDynamics(config)
    print(dynamics.activation)
    print(dynamics.write_derived_params)
    print(dynamics.write_initial_state)
    print(dynamics.write_state_velocity)
    print(dynamics.currents)