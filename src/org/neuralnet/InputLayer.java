package org.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InputLayer extends Layer {
    public InputLayer(int neuronsCount, int inputsNumber) {
        List<Double> weightsInput;

        neurons = new ArrayList<>();

        for (int i = 0; i < neuronsCount + 1; i++) {
            Neuron neuron = new Neuron();
            weightsInput = new ArrayList<>();
            for (int j = 0; j < inputsNumber; j++) {
                weightsInput.add(neuron.initNeuron());
            }
            neuron.setWeightsInput(weightsInput);
            neurons.add(neuron);
        }
    }

    @Override
    public void printLayer() {
        super.printLayer();
        System.out.println("### INPUT LAYER ###");
        int n = 1;
        for (Neuron neuron : neurons) {
            System.out.println("Neuron #" + n + ":");
            System.out.println("Input Weights:");
            System.out.println(Arrays.deepToString(neuron.getWeightsInput().toArray()));
            n++;
        }
    }
}
