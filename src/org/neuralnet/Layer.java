package org.neuralnet;

import java.util.List;

public abstract class Layer {
    protected List<Neuron> neurons;

    public void printLayer() {

    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public void setNeurons(List<Neuron> neurons) {
        this.neurons = neurons;
    }

    public int getNeuronsCount() {
        return this.neurons.size();
    }
}
