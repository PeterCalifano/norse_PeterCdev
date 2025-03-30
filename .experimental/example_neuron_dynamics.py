from ast import TypeVar
from functools import partial
from ipywidgets import interact, IntSlider, FloatSlider
import matplotlib.pyplot as plt
import torch
import norse
import numpy as np
from norse.torch.module.snn import SNNCell, SNN, SNNRecurrent, SNNRecurrentCell 


#plt.style.use("../resources/matplotlibrc")
norse.__version__


def visualize_activation_response(neuron_type: type[SNNCell | SNN | SNNRecurrent | SNNRecurrentCell], **kwargs : dict):
    valid_kwargs = {key: value for key, value in kwargs.items() if key in neuron_type.__init__.__annotations__}
    activation = neuron_type(**valid_kwargs) # type: ignore
    print('Activation function: ', activation) 

    # Generate a sample input tensor
    sample_data = torch.zeros(1000, 1)
    sample_data[20] = 1.0 # Spike at 20 ms
    sample_data[100] = 1.0 # Spike at 100 ms

    # Pass into stateful activation function (integrator)
    # The LI activation function will return a tuple of (voltage trace, neuron state)
    voltage_trace, _ = activation(sample_data)

    # Plot the voltage trace
    plt.xlabel('time [ms]')
    plt.ylabel('membrane potential')
    plt.plot(voltage_trace.detach())
    plt.axvline(20, color='red')
    plt.axvline(100, color='red')

# Define interactive widgets for parameters (slider)
IntSlider = partial(IntSlider, continuous_update=False)
FloatSlider = partial(FloatSlider, continuous_update=False)
# Requires jupyter notebook for interactivity

@interact(
    tau_mem=FloatSlider(min=10, max=200, step=1.0, value=10),
    tau_syn=FloatSlider(min=10, max=200, step=1.0, value=20),
    t0=IntSlider(min=0, max=1000, step=1, value=20),
    t1=IntSlider(min=0, max=1000, step=1, value=100),
)

# Define experiment with slider to visualize how dynamics change with different parameters
def experiment_LI(tau_mem, tau_syn, t0, t1):
    plt.figure()

    num_neurons = 1
    tau_mem_inv = torch.tensor([1/(tau_mem * 0.001)])
    tau_syn_inv = torch.tensor([1/(tau_syn * 0.001)])
    data = torch.zeros(1000, num_neurons)
    data[t0] = 1.0
    data[t1] = 1.0

    voltage_trace, _ = norse.torch.LI(p=norse.torch.LIParameters(
        tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv))(data)
    
    plt.xlabel('time [ms]')
    plt.ylabel('membrane potential')
    for i in range(num_neurons):
        plt.plot(voltage_trace.detach()[:, i])
    plt.axvline(t0, color='red', alpha=0.9)
    plt.axvline(t1, color='red', alpha=0.9)
    plt.show()

# Helper function to track voltage of neuron since cell returns spike outputs only by default
def integrate_and_record_voltages(cell):
    def integrate(input_spike_train):
        T = input_spike_train.shape[0]
        s = None
        spikes = []
        voltage_trace = []
        for ts in range(T):
            z, s = cell(input_spike_train[ts], s)
            spikes.append(z)
            voltage_trace.append(s.v)
        return torch.stack(spikes), torch.stack(voltage_trace)
    return integrate


def experiment_LIF(tau_mem, tau_syn, v_th, t0, t1):
    plt.figure()
    num_neurons = 1
    tau_syn_inv = torch.tensor([1/(tau_syn * 0.001)])
    tau_mem_inv = torch.tensor([1/(tau_mem * 0.001)])
    data = torch.zeros(1000, num_neurons)
    data[20] = 1.0
    data[t0] = 1.0
    data[t1] = 1.0

    cell = norse.torch.LIFCell(p=norse.torch.LIFParameters(
        tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv, v_th=torch.as_tensor(v_th)))
    lif_integrate = integrate_and_record_voltages(cell)

    voltage_trace, _ = norse.torch.LI(p=norse.torch.LIParameters(
        tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv))(data)
    zs, lif_voltage_trace = lif_integrate(data)
    plt.xlabel('time [ms]')
    plt.ylabel('membrane potential')
    plt.plot(voltage_trace.detach(), label="LI")
    plt.plot(lif_voltage_trace.detach(), label="LIF")
    plt.axhline(v_th, color='grey')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Visualize activation response for different neuron types
    #visualize_activation_response(norse.torch.LI)

    # Run the experiment with default parameters
    #experiment_LI(tau_mem=10, tau_syn=20, t0=20, t1=500)

    # Run the experiment with LIF neuron type
    experiment_LIF(tau_mem=10, tau_syn=20, v_th=0.5, t0=20, t1=500)