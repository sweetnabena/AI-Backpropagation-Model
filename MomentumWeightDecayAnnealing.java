package data;

import java.util.*;

public class MomentumWeightDecayAnnealing {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 14;

    private static final double INITIAL_LEARNING_RATE = 0.1;
    private static final double MOMENTUM = 0.9;
    private static final double DECAY_RATE = 0.0005;

    private static final double P = 0.01;
    private static final double Q = 0.5;
    private static final double R = 48000.0;

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;

    private double[][] prevDeltaInputHidden;
    private double[] prevDeltaHiddenOutput;
    private double[] prevDeltaHiddenBiases;
    private double prevDeltaOutputBias;

    private double outputMin;
    private double outputMax;

    public MomentumWeightDecayAnnealing() {
        Random rand = new Random(42);

        weightsInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        weightsHiddenOutput = new double[HIDDEN_NODES];
        hiddenBiases = new double[HIDDEN_NODES];
        outputBias = rand.nextDouble() - 0.5;

        prevDeltaInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        prevDeltaHiddenOutput = new double[HIDDEN_NODES];
        prevDeltaHiddenBiases = new double[HIDDEN_NODES];
        prevDeltaOutputBias = 0;

        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() - 0.5;
            }
        }

        for (int j = 0; j < HIDDEN_NODES; j++) {
            weightsHiddenOutput[j] = rand.nextDouble() - 0.5;
            hiddenBiases[j] = rand.nextDouble() - 0.5;
        }
    }

    public void setOutputMinMax(double min, double max) {
        this.outputMin = min;
        this.outputMax = max;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private double computeWeightDecayPenalty() {
        double sum = 0;
        for (double[] row : weightsInputHidden)
            for (double w : row) sum += w * w;
        for (double w : weightsHiddenOutput)
            sum += w * w;
        return sum / 2.0;
    }

    private double computeAnnealedLearningRate(int epoch) {
        return P + (Q - P) * (1 - 1.0 / (1 + Math.exp((10 - 20.0 * epoch / R))));
    }

    public void train(double[][] trainingInputs, double[] expectedOutputs, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues,
                      List<Double> learningRateTrack) {
        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;
            double learningRate = computeAnnealedLearningRate(epoch);
            learningRateTrack.add(learningRate);

            for (int t = 0; t < trainingInputs.length; t++) {
                double[] inputs = trainingInputs[t];
                double expected = expectedOutputs[t];

                double[] hiddenOutputs = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double sum = hiddenBiases[j];
                    for (int i = 0; i < INPUT_NODES; i++) {
                        sum += inputs[i] * weightsInputHidden[i][j];
                    }
                    hiddenOutputs[j] = sigmoid(sum);
                }

                double outputSum = outputBias;
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    outputSum += hiddenOutputs[j] * weightsHiddenOutput[j];
                }
                double predicted = sigmoid(outputSum);

                if (epoch == epochs) {
                    actualValues.add(denormalize(expected));
                    predictedValues.add(denormalize(predicted));
                }

                double error = expected - predicted;
                double omega = computeWeightDecayPenalty();
                double deltaOutput = (error + DECAY_RATE * omega) * sigmoidDerivative(predicted);
                totalError += error * error + DECAY_RATE * omega;

                double[] deltaHidden = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * sigmoidDerivative(hiddenOutputs[j]);
                }

                // Hidden to output with momentum
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double delta = learningRate * deltaOutput * hiddenOutputs[j] + MOMENTUM * prevDeltaHiddenOutput[j];
                    weightsHiddenOutput[j] += delta;
                    prevDeltaHiddenOutput[j] = delta;
                }

                double deltaOutBias = learningRate * deltaOutput + MOMENTUM * prevDeltaOutputBias;
                outputBias += deltaOutBias;
                prevDeltaOutputBias = deltaOutBias;

                // Input to hidden with momentum
                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        double delta = learningRate * deltaHidden[j] * inputs[i] + MOMENTUM * prevDeltaInputHidden[i][j];
                        weightsInputHidden[i][j] += delta;
                        prevDeltaInputHidden[i][j] = delta;
                    }
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double delta = learningRate * deltaHidden[j] + MOMENTUM * prevDeltaHiddenBiases[j];
                    hiddenBiases[j] += delta;
                    prevDeltaHiddenBiases[j] = delta;
                }
            }

            trainingLosses.add(totalError / trainingInputs.length);
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.10f - LR: %.5f%n", epoch, totalError / trainingInputs.length, learningRate);
            }
        }
    }

    public double predict(double[] input) {
        double[] hiddenOutputs = new double[HIDDEN_NODES];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            double sum = hiddenBiases[j];
            for (int i = 0; i < INPUT_NODES; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            hiddenOutputs[j] = sigmoid(sum);
        }

        double outputSum = outputBias;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            outputSum += hiddenOutputs[j] * weightsHiddenOutput[j];
        }

        return denormalize(sigmoid(outputSum));
    }

    private double denormalize(double normVal) {
        return (normVal - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
    }

    public void setOutputRange(double min, double max) {
        this.outputMin = min;
        this.outputMax = max;
    }
}
