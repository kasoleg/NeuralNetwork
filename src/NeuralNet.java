import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    private InputLayer inputLayer;
    private List<HiddenLayer> hiddenLayers;
    private OutputLayer outputLayer;

    private double learningRate;
    private ActivationFunctions activationFunction;

    public void init() {
        inputLayer = new InputLayer(2, 1);
        HiddenLayer hiddenLayer1 = new HiddenLayer(3, 2, 3);
        HiddenLayer hiddenLayer2 = new HiddenLayer(3, 3, 1);
        hiddenLayers = new ArrayList<>();
        hiddenLayers.add(hiddenLayer1);
        hiddenLayers.add(hiddenLayer2);
        outputLayer = new OutputLayer(1, 1);
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

    public void printNet() {
        inputLayer.printLayer();
        System.out.println();
        System.out.println("### HIDDEN LAYER ###");
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
