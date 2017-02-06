package org.neuralnet;

import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    private InputLayer inputLayer;
    private List<HiddenLayer> hiddenLayers;
    private OutputLayer outputLayer;

    private double learningRate;
    private double targetError;
    private double trainingError;
    private ActivationFunctions activationFunction;
    private ActivationFunctions activationFunctionOutputLayer;
    private double[][] trainSet;
    private double[] realOutputSet;
    private double[][] realMatrixOutputSet;
    private ArrayList<Double> listOfMSE = new ArrayList<>();
    private TrainingTypes trainType;
    private double errorMean;

    public void init(int numberOfInputNeurons, int numberOfHiddenLayers, int numberOfNeuronsInHiddenLayer, int numberOfOutputNeurons) {
        inputLayer = new InputLayer(numberOfInputNeurons, 1);
        //HiddenLayer hiddenLayer1 = new HiddenLayer(3, 2, 3); 3 - количество нейронов 2 - количество нейронов на предыдущем слое 3 - количество нейронов на следующем слое
        //HiddenLayer hiddenLayer2 = new HiddenLayer(3, 3, 1);
        hiddenLayers = new ArrayList<>();
        for (int i = 0; i < numberOfHiddenLayers; i++) {
            if (i != (numberOfHiddenLayers - 1)) {
                hiddenLayers.add(new HiddenLayer(numberOfNeuronsInHiddenLayer, numberOfNeuronsInHiddenLayer, numberOfNeuronsInHiddenLayer));
            } else {
                hiddenLayers.add(new HiddenLayer(numberOfNeuronsInHiddenLayer, numberOfNeuronsInHiddenLayer, numberOfOutputNeurons));
            }
        }
        outputLayer = new OutputLayer(numberOfOutputNeurons, 1);
    }

    public InputLayer getInputLayer() {
        return inputLayer;
    }

    public void setInputLayer(InputLayer inputLayer) {
        this.inputLayer = inputLayer;
    }

    public List<HiddenLayer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(List<HiddenLayer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    public void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public ActivationFunctions getActivationFunction() {
        return activationFunction;
    }

    public void setTrainSet(double[][] trainSet) {
        this.trainSet = trainSet;
    }

    public void setRealOutputSet(double[] realOutputSet) {
        this.realOutputSet = realOutputSet;
    }

    public void setTargetError(double targetError) {
        this.targetError = targetError;
    }

    public void setTrainingError(double trainingError) {
        this.trainingError = trainingError;
    }

    public void setActivationFunction(ActivationFunctions activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setTrainType(TrainingTypes trainType) {
        this.trainType = trainType;
    }

    public double[][] getTrainSet() {
        return trainSet;
    }

    public double[] getRealOutputSet() {
        return realOutputSet;
    }

    public double getTargetError() {
        return targetError;
    }

    public ArrayList<Double> getListOfMSE() {
        return listOfMSE;
    }

    public TrainingTypes getTrainType() {
        return trainType;
    }

    public double getErrorMean() {
        return errorMean;
    }

    public ActivationFunctions getActivationFunctionOutputLayer() {
        return activationFunctionOutputLayer;
    }

    public void setActivationFunctionOutputLayer(ActivationFunctions activationFunctionOutputLayer) {
        this.activationFunctionOutputLayer = activationFunctionOutputLayer;
    }

    public double[][] getRealMatrixOutputSet() {
        return realMatrixOutputSet;
    }

    public void setRealMatrixOutputSet(double[][] realMatrixOutputSet) {
        this.realMatrixOutputSet = realMatrixOutputSet;
    }

    public void setErrorMean(double errorMean) {
        this.errorMean = errorMean;
    }

    public void print() {
        inputLayer.printLayer();
        if (hiddenLayers.size() > 0) {
            System.out.println();
            System.out.println("### HIDDEN LAYER ###");
        }
        int h = 1;
        for (HiddenLayer hiddenLayer : hiddenLayers) {
            System.out.println("Hidden Layer #" + h);
            hiddenLayer.printLayer();
            h++;
        }
        System.out.println();
        outputLayer.printLayer();
    }
}
