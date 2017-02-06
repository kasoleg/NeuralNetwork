package org.neuralnet;

import java.util.List;
import java.util.Random;

public class Neuron {
    private List<Double> weightsInput;
    private List<Double> weightsOutput;
    private double outputValue;
    private double error;
    private double sensibility;

    public double initNeuron() {
        Random random = new Random();
        return random.nextDouble();
    }

    public List<Double> getWeightsInput() {
        return weightsInput;
    }

    public void setWeightsInput(List<Double> weightsInput) {
        this.weightsInput = weightsInput;
    }

    public List<Double> getWeightsOutput() {
        return weightsOutput;
    }

    public void setWeightsOutput(List<Double> weightsOutput) {
        this.weightsOutput = weightsOutput;
    }

    public void setOutputValue(double outputValue) {
        this.outputValue = outputValue;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getSensibility() {
        return sensibility;
    }

    public void setSensibility(double sensibility) {
        this.sensibility = sensibility;
    }
}
