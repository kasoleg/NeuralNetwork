import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Main test = new Main();
        test.testPerceptron();
        test.testAdaline();
    }

    public void testPerceptron() {
        NeuralNet net = new NeuralNet();
        net.init(2, 0, 0, 1);

        System.out.println("---------PERCEPTRON INIT NET---------");

        net.print();

        net.setTrainSet(new double[][]{{1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}});
        net.setRealOutputSet(new double[]{0.0, 0.0, 0.0, 1.0});
        net.setTargetError(0.002);
        net.setLearningRate(1.0);
        net.setTrainType(TrainingTypes.PERCEPTRON);
        net.setActivationFunction(ActivationFunctions.STEP);

        Perceptron perceptron = new Perceptron();
        perceptron.train(net, 10);

        System.out.println();
        System.out.println("---------PERCEPTRON TRAINED NET---------");

        net.print();

        System.out.println();
        System.out.println("---------PERCEPTRON PRINT RESULT---------");

        perceptron.printTrainedNetResult(net);
    }

    public void testAdaline() {
        NeuralNet net = new NeuralNet();

        net.init(3, 0, 0, 1);

        System.out.println("---------ADALINE INIT NET---------");

        net.print();

        // first column has BIAS
        net.setTrainSet(new double[][]{{1.0, 0.98, 0.94, 0.95},
                {1.0, 0.60, 0.60, 0.85}, {1.0, 0.35, 0.15, 0.15},
                {1.0, 0.25, 0.30, 0.98}, {1.0, 0.75, 0.85, 0.91},
                {1.0, 0.43, 0.57, 0.87}, {1.0, 0.05, 0.06, 0.01}});
        net.setRealOutputSet(new double[]{0.80, 0.59, 0.23, 0.45, 0.74,
                0.63, 0.10});
        net.setTargetError(0.0001);
        net.setLearningRate(0.5);
        net.setTrainType(TrainingTypes.ADALINE);
        net.setActivationFunction(ActivationFunctions.LINEAR);

        Adaline adaline = new Adaline();
        adaline.train(net, 10);

        System.out.println();
        System.out.println("---------ADALINE TRAINED NET---------");

        net.print();

        System.out.println();
        System.out.println("---------ADALINE PRINT RESULT---------");

        adaline.printTrainedNetResult(net);

        System.out.println();
        System.out.println("---------ADALINE MSE BY EPOCH---------");
        System.out.println(Arrays.deepToString(net.getListOfMSE().toArray()).replace(" ", "\n"));
    }
}
