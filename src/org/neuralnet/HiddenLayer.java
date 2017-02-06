package org.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HiddenLayer extends Layer {
    public HiddenLayer(int neuronsCount, int inputsNumber, int outputsNumber) {
        List<Double> weightsInput;
        List<Double> weightsOutput;
        neurons = new ArrayList<>();

        for (int i = 0; i <= neuronsCount; i++) {
            Neuron neuron = new Neuron();
            weightsInput = new ArrayList<>();
            weightsOutput = new ArrayList<>();
            if (i > 0) {
                for (int j = 0; j < inputsNumber; j++) {
                    weightsInput.add(neuron.initNeuron());
                }
            }
            neuron.setWeightsInput(weightsInput);
            for (int j = 0; j < outputsNumber; j++) {
                weightsOutput.add(neuron.initNeuron());
            }
            neuron.setWeightsOutput(weightsOutput);
            neurons.add(neuron);
        }
    }

    @Override
    public void printLayer() {
        super.printLayer();
        int n = 1;
        for (Neuron neuron : neurons) {
            System.out.println("Neuron #" + n);
            System.out.println("Input Weights:");
            System.out.println(Arrays.deepToString(neuron.getWeightsInput().toArray()));
            System.out.println("Output Weights:");
            System.out.println(Arrays.deepToString(neuron.getWeightsOutput().toArray()));
            n++;
        }
    }
}
