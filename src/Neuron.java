import java.util.List;
import java.util.Random;

public class Neuron {
    private List<Double> weightsInput;
    private List<Double> weightsOutput;

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
}
