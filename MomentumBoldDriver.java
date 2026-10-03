package data;

import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.time.Day;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class MomentumBoldDriver {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 14;
    private static final double MOMENTUM = 0.9;
    private static final double MIN_LEARNING_RATE = 0.01;
    private static final double MAX_LEARNING_RATE = 0.5;

    private double learningRate = 0.1;

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

    public void setOutputMinMax(double min, double max) {
        this.outputMin = min;
        this.outputMax = max;
    }

    private double denormalize(double normalizedValue) {
        return (normalizedValue - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
    }

    public MomentumBoldDriver() {
        Random rand = new Random(42);
        weightsInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        weightsHiddenOutput = new double[HIDDEN_NODES];
        hiddenBiases = new double[HIDDEN_NODES];

        prevDeltaInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        prevDeltaHiddenOutput = new double[HIDDEN_NODES];
        prevDeltaHiddenBiases = new double[HIDDEN_NODES];
        prevDeltaOutputBias = 0;

        outputBias = rand.nextDouble() - 0.5;

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

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public void train(double[][] inputs, double[] targets, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues) {

        double previousError = Double.MAX_VALUE;

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;

            double[][] prevWeightsIH = deepCopy(weightsInputHidden);
            double[] prevWeightsHO = Arrays.copyOf(weightsHiddenOutput, weightsHiddenOutput.length);
            double[] prevBiases = Arrays.copyOf(hiddenBiases, hiddenBiases.length);
            double prevOutBias = outputBias;

            for (int t = 0; t < inputs.length; t++) {
                double[] input = inputs[t];
                double target = targets[t];

                double[] hiddenOutputs = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double sum = hiddenBiases[j];
                    for (int i = 0; i < INPUT_NODES; i++) {
                        sum += input[i] * weightsInputHidden[i][j];
                    }
                    hiddenOutputs[j] = sigmoid(sum);
                }

                double finalSum = outputBias;
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    finalSum += hiddenOutputs[j] * weightsHiddenOutput[j];
                }
                double finalOutput = sigmoid(finalSum);

                if (epoch == epochs) {
                    actualValues.add(denormalize(target));
                    predictedValues.add(denormalize(finalOutput));
                }

                double outputError = target - finalOutput;
                double deltaOutput = outputError * sigmoidDerivative(finalOutput);

                double[] deltaHidden = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * sigmoidDerivative(hiddenOutputs[j]);
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double delta = learningRate * deltaOutput * hiddenOutputs[j] + MOMENTUM * prevDeltaHiddenOutput[j];
                    weightsHiddenOutput[j] += delta;
                    prevDeltaHiddenOutput[j] = delta;
                }

                double deltaBiasOut = learningRate * deltaOutput + MOMENTUM * prevDeltaOutputBias;
                outputBias += deltaBiasOut;
                prevDeltaOutputBias = deltaBiasOut;

                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        double delta = learningRate * deltaHidden[j] * input[i] + MOMENTUM * prevDeltaInputHidden[i][j];
                        weightsInputHidden[i][j] += delta;
                        prevDeltaInputHidden[i][j] = delta;
                    }
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double delta = learningRate * deltaHidden[j] + MOMENTUM * prevDeltaHiddenBiases[j];
                    hiddenBiases[j] += delta;
                    prevDeltaHiddenBiases[j] = delta;
                }

                totalError += outputError * outputError;
            }

            double avgError = totalError / inputs.length;
            trainingLosses.add(avgError);

            if (epoch % 1000 == 0 && epoch > 0) {
                double errorChange = (avgError - previousError) / previousError;

                if (errorChange > 0.04) {
                    weightsInputHidden = deepCopy(prevWeightsIH);
                    weightsHiddenOutput = Arrays.copyOf(prevWeightsHO, prevWeightsHO.length);
                    hiddenBiases = Arrays.copyOf(prevBiases, prevBiases.length);
                    outputBias = prevOutBias;
                    learningRate = Math.max(MIN_LEARNING_RATE, learningRate * 0.7);
                } else if (errorChange < 0) {
                    learningRate = Math.min(MAX_LEARNING_RATE, learningRate * 1.05);
                }
                previousError = avgError;
            }

            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.10f - Learning Rate: %.5f\n", epoch, avgError, learningRate);
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

        double finalSum = outputBias;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            finalSum += hiddenOutputs[j] * weightsHiddenOutput[j];
        }
        return denormalize(sigmoid(finalSum));
    }

    private double[][] deepCopy(double[][] original) {
        double[][] copy = new double[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            copy[i] = Arrays.copyOf(original[i], original[i].length);
        }
        return copy;
    }
}
