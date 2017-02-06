package org.neuralnet.learn;

import org.neuralnet.*;

import java.util.ArrayList;
import java.util.List;

public abstract class Training {
    private double error;
    protected double mse;

    public NeuralNet train(NeuralNet n, int epochs) {
        List<Double> inputWeights;

        int rows = n.getTrainSet().length;
        int cols = n.getTrainSet()[0].length;

        int epoch = 0;
        while (epoch < epochs) {

            double estimatedOutput = 0.0;
            double realOutput = 0.0;

            for (int i = 0; i < rows; i++) {

                double netValue = 0.0;

                for (int j = 0; j < cols; j++) {
                    inputWeights = n.getInputLayer().getNeurons().get(j).getWeightsInput();
                    double inputWeight = inputWeights.get(0);
                    netValue = netValue + inputWeight * n.getTrainSet()[i][j];
                }

                estimatedOutput = this.activationFunction(n.getActivationFunction(), netValue);
                realOutput = n.getRealOutputSet()[i];

                error = realOutput - estimatedOutput;

                // System.out.println("Epoch: " + epoch + " / Error: " + error);

                if (Math.abs(error) > n.getTargetError()) {
                    // fix weights
                    InputLayer inputLayer = new InputLayer(2, 1);
                    inputLayer.setNeurons(teachNeuronsOfLayer(cols,
                            i, n, netValue));
                    n.setInputLayer(inputLayer);
                }

            }

            mse = Math.pow(realOutput - estimatedOutput, 2.0);
            n.getListOfMSE().add(mse);

            epoch++;

        }

        n.setTrainingError(error);
        return n;
    }

    private List<Neuron> teachNeuronsOfLayer(int numberOfInputNeurons,
                                             int line, NeuralNet n, double netValue) {
        List<Neuron> listOfNeurons = new ArrayList<>();
        List<Double> inputWeightsNew = new ArrayList<>();
        List<Double> inputWeightsOld;

        for (int j = 0; j < numberOfInputNeurons; j++) {
            inputWeightsOld = n.getInputLayer().getNeurons().get(j).getWeightsInput();
            double inputWeightOld = inputWeightsOld.get(0);

            inputWeightsNew.add(calcNewWeight(n.getTrainType(), inputWeightOld, n, error, n.getTrainSet()[line][j], netValue));

            Neuron neuron = new Neuron();
            neuron.setWeightsInput(inputWeightsNew);
            listOfNeurons.add(neuron);
            inputWeightsNew = new ArrayList<>();
        }

        return listOfNeurons;

    }

    private double calcNewWeight(TrainingTypes trainType, double inputWeightOld, NeuralNet n, double error, double trainSample, double netValue) {
        switch (trainType) {
            case PERCEPTRON:
                return inputWeightOld + n.getLearningRate() * error * trainSample;
            case ADALINE:
                return inputWeightOld + n.getLearningRate() * error * trainSample * derivativeActivationFnc(n.getActivationFunction(), netValue);
            default:
                throw new IllegalArgumentException(trainType + " does not exist in TrainingTypes");
        }
    }

    public double derivativeActivationFnc(ActivationFunctions function, double value) {
        switch (function) {
            case LINEAR:
                return derivativeFunctionLinear(value);
            case SIGLOG:
                return derivativeFunctionSigLog(value);
            case HYPERTAN:
                return derivativeFunctionHyperTan(value);
            default:
                throw new IllegalArgumentException(function + " does not exist in ActivationFunctions");
        }
    }

    private double derivativeFunctionLinear(double v) {
        return 1.0;
    }

    private double derivativeFunctionSigLog(double v) {
        return v * (1.0 - v);
    }

    private double derivativeFunctionHyperTan(double v) {
        return (1.0 / Math.pow(Math.cosh(v), 2.0));
    }

    protected double activationFunction(ActivationFunctions function, double value) {
        switch (function) {
            case STEP:
                return functionStep(value);
            case LINEAR:
                return functionLinear(value);
            case SIGLOG:
                return functionSigLog(value);
            case HYPERTAN:
                return functionHyperTan(value);
            default:
                throw new IllegalArgumentException(function + " does not exist in ActivationFunctions");
        }
    }

    private double functionStep(double v) {
        if (v >= 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    private double functionLinear(double v) {
        return v;
    }

    private double functionSigLog(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    private double functionHyperTan(double v) {
        return Math.tanh(v);
    }

    public void printTrainedNetResult(NeuralNet trainedNet) {

        int rows = trainedNet.getTrainSet().length;
        int cols = trainedNet.getTrainSet()[0].length;

        List<Double> inputWeightIn;

        for (int i = 0; i < rows; i++) {
            double netValue = 0.0;
            for (int j = 0; j < cols; j++) {
                inputWeightIn = trainedNet.getInputLayer().getNeurons().get(j).getWeightsInput();
                double inputWeight = inputWeightIn.get(0);
                netValue = netValue + inputWeight * trainedNet.getTrainSet()[i][j];

                System.out.print(trainedNet.getTrainSet()[i][j] + "\t");
            }

            double estimatedOutput = activationFunction(trainedNet.getActivationFunction(), netValue);

            System.out.print(" NET OUTPUT: " + estimatedOutput + "\t");
            System.out.print(" REAL OUTPUT: " + trainedNet.getRealOutputSet()[i] + "\t");
            double error = estimatedOutput - trainedNet.getRealOutputSet()[i];
            System.out.print(" ERROR: " + error + "\n");

        }

    }
}
