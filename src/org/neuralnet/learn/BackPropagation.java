package org.neuralnet.learn;

import org.neuralnet.HiddenLayer;
import org.neuralnet.NeuralNet;
import org.neuralnet.Neuron;

import java.util.List;

public class BackPropagation extends Training {
    @Override
    public NeuralNet train(NeuralNet n, int epochs) {
        mse = 1;

        int epoch = 0;
        while (mse > n.getTargetError() && epoch < epochs) {
            int rows = n.getTrainSet().length;
            double sumErrors = 0;

            for (int i = 0; i < rows; i++) {
                forward(n, i);
                backPropagation(n, i);
                sumErrors = sumErrors + n.getErrorMean();
            }

            mse = sumErrors / rows;

            System.out.println(mse);

            epoch++;
        }

        System.out.println("Number of epochs: " + epoch);

        return n;
    }

    protected NeuralNet forward(NeuralNet n, int row) {

        List<HiddenLayer> hiddenLayers = n.getHiddenLayers();

        double estimatedOutput = 0.0;
        double realOutput = 0.0;
        double sumError = 0.0;

        if (hiddenLayers.size() > 0) {

            int i = 0;

            for (HiddenLayer hiddenLayer : hiddenLayers) {
                for (Neuron neuron : hiddenLayer.getNeurons()) {
                    double netValueOut = 0.0;

                    if (neuron.getWeightsInput().size() > 0) { //exclude bias
                        double netValue = 0.0;

                        for (int j = 0; j < hiddenLayer.getNeuronsCount() - 1; j++) { //exclude bias
                            double hiddenWeightInput = neuron.getWeightsInput().get(j);
                            netValue = netValue + hiddenWeightInput * n.getTrainSet()[row][j];
                        }

                        //output hidden layer (1)
                        netValueOut = super.activationFunction(n.getActivationFunction(), netValue);
                        neuron.setOutputValue(netValueOut);
                    } else {
                        neuron.setOutputValue(1.0);
                    }

                }


                //output hidden layer (2)
                for (int j = 0; j < n.getOutputLayer().getNeuronsCount(); j++) {
                    double netValue = 0.0;
                    double netValueOut = 0.0;

                    for (Neuron neuron : hiddenLayer.getNeurons()) {
                        double hiddenWeightOutput = neuron.getWeightsOutput().get(j);
                        netValue = netValue + hiddenWeightOutput * neuron.getOutputValue();
                    }

                    netValueOut = activationFunction(n.getActivationFunctionOutputLayer(), netValue);

                    n.getOutputLayer().getNeurons().get(j).setOutputValue(netValueOut);

                    //error
                    estimatedOutput = netValueOut;
                    realOutput = n.getRealMatrixOutputSet()[row][j];
                    double error = realOutput - estimatedOutput;
                    n.getOutputLayer().getNeurons().get(j).setError(error);
                    sumError = sumError + Math.pow(error, 2.0);

                }

                //error mean
                double errorMean = sumError / n.getOutputLayer().getNeuronsCount();
                n.setErrorMean(errorMean);

                n.getHiddenLayers().get(i).setNeurons(hiddenLayer.getNeurons());

                i++;

            }

        }

        return n;

    }

    protected NeuralNet backPropagation(NeuralNet n, int row) {
        List<Neuron> outputLayer = n.getOutputLayer().getNeurons();
        List<Neuron> hiddenLayer = n.getHiddenLayers().get(0).getNeurons();

        double error = 0.0;
        double netValue = 0.0;
        double sensibility = 0.0;

        //sensibility output layer
        for (Neuron neuron : outputLayer) {
            error = neuron.getError();
            netValue = neuron.getOutputValue();
            sensibility = derivativeActivationFnc(n.getActivationFunctionOutputLayer(), netValue) * error;

            neuron.setSensibility(sensibility);
        }

        n.getOutputLayer().setNeurons(outputLayer);


        //sensibility hidden layer
        for (Neuron neuron : hiddenLayer) {
            sensibility = 0.0;

            if (neuron.getWeightsInput().size() > 0) { //exclude bias
                List<Double> listOfWeightsOut = neuron.getWeightsOutput();
                double tempSensibility = 0.0;

                int i = 0;
                for (Double weight : listOfWeightsOut) {
                    tempSensibility += weight * outputLayer.get(i).getSensibility();
                    i++;
                }

                sensibility = derivativeActivationFnc(n.getActivationFunction(), neuron.getOutputValue()) * tempSensibility;

                neuron.setSensibility(sensibility);

            }

        }

        //fix weights (teach) [output layer to hidden layer]
        for (int i = 0; i < n.getOutputLayer().getNeuronsCount(); i++) {
            for (Neuron neuron : hiddenLayer) {
                double newWeight = neuron.getWeightsOutput().get(i) + (n.getLearningRate() * outputLayer.get(i).getSensibility() * neuron.getOutputValue());
                neuron.getWeightsOutput().set(i, newWeight);
            }

        }

        //fix weights (teach) [hidden layer to input layer]
        for (Neuron neuron : hiddenLayer) {
            List<Double> hiddenLayerInputWeights = neuron.getWeightsInput();

            if (hiddenLayerInputWeights.size() > 0) { //exclude bias
                int i = 0;
                double newWeight = 0.0;
                for (int j = 0; j < n.getInputLayer().getNeuronsCount(); j++) {
                    newWeight = hiddenLayerInputWeights.get(i) + (n.getLearningRate() * neuron.getSensibility() * n.getTrainSet()[row][j]);
                    neuron.getWeightsInput().set(i, newWeight);
                    i++;
                }

            }

        }

        n.getHiddenLayers().get(0).setNeurons(hiddenLayer);

        return n;

    }
}
