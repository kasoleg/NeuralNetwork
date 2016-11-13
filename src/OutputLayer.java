import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class OutputLayer extends Layer {
    public OutputLayer(int neuronsCount, int outputsNumber) {
        List<Double> weightsOutput;
        neurons = new ArrayList<>();

        for (int i = 0; i < neuronsCount; i++) {
            Neuron neuron = new Neuron();
            weightsOutput = new ArrayList<>();
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
        System.out.println("### OUTPUT LAYER ###");
        int n = 1;
        for (Neuron neuron : neurons) {
            System.out.println("Neuron #" + n + ":");
            System.out.println("Output Weights:");
            System.out.println(Arrays.deepToString(neuron.getWeightsOutput().toArray()));
            n++;
        }
    }
}
