package data;

import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.JFrame;

import org.jfree.chart.*;
import org.jfree.chart.axis.*;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.time.*;
import org.jfree.data.xy.*;

public class Momentum {

    // Define network structure and parameters
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 5;
    private static final double LEARNING_RATE = 0.1;
    private static final double MOMENTUM = 0.9;

    // Weights and biases
    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;

    // Momentum-related variables to store previous deltas
    private double[][] prevDeltaInputHidden;
    private double[] prevDeltaHiddenOutput;
    private double[] prevDeltaHiddenBiases;
    private double prevDeltaOutputBias;

    // For denormalisation of outputs
    private double outputMin;
    private double outputMax;

    // Set original output min and max values for denormalisation later
    public void setOutputMinMax(double min, double max) {
        this.outputMin = min;
        this.outputMax = max;
    }

    // Convert normalised value back to original range
    private double denormalize(double normalizedValue) {
        return (normalizedValue - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
    }

    // Constructor: Initialise weights and momentum storage
    public Momentum() {
        Random rand = new Random(42);

        // Weight initialisation
        weightsInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        weightsHiddenOutput = new double[HIDDEN_NODES];
        hiddenBiases = new double[HIDDEN_NODES];

        // Initialise momentum storage with zeros
        prevDeltaInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        prevDeltaHiddenOutput = new double[HIDDEN_NODES];
        prevDeltaHiddenBiases = new double[HIDDEN_NODES];
        prevDeltaOutputBias = 0;

        // Random initial bias for output node
        outputBias = rand.nextDouble() - 0.5;

        // Random weights for input to hidden
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() - 0.5;
            }
        }

        // Random weights and biases for hidden to output
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weightsHiddenOutput[j] = rand.nextDouble() - 0.5;
            hiddenBiases[j] = rand.nextDouble() - 0.5;
        }
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid used for gradient calculation
    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    // Train the network using backpropagation with momentum
    public void train(double[][] inputs, double[] targets, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues) {

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;

            for (int t = 0; t < inputs.length; t++) {
                double[] input = inputs[t];
                double target = targets[t];

                // ===== Forward Pass =====
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

                // Store predictions on the last epoch for analysis
                if (epoch == epochs) {
                    actualValues.add(denormalize(target));
                    predictedValues.add(denormalize(finalOutput));
                }

                // ===== Backward Pass =====
                double outputError = target - finalOutput;
                double deltaOutput = outputError * sigmoidDerivative(finalOutput);

                double[] deltaHidden = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * sigmoidDerivative(hiddenOutputs[j]);
                }

                // ===== Weight Updates with Momentum =====

                // Update weights between hidden and output layer
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double delta = LEARNING_RATE * deltaOutput * hiddenOutputs[j] + MOMENTUM * prevDeltaHiddenOutput[j];
                    weightsHiddenOutput[j] += delta;
                    prevDeltaHiddenOutput[j] = delta;
                }

                // Update output bias
                double deltaBiasOut = LEARNING_RATE * deltaOutput + MOMENTUM * prevDeltaOutputBias;
                outputBias += deltaBiasOut;
                prevDeltaOutputBias = deltaBiasOut;

                // Update weights between input and hidden layer
                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        double delta = LEARNING_RATE * deltaHidden[j] * input[i] + MOMENTUM * prevDeltaInputHidden[i][j];
                        weightsInputHidden[i][j] += delta;
                        prevDeltaInputHidden[i][j] = delta;
                    }
                }

                // Update hidden biases
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    double delta = LEARNING_RATE * deltaHidden[j] + MOMENTUM * prevDeltaHiddenBiases[j];
                    hiddenBiases[j] += delta;
                    prevDeltaHiddenBiases[j] = delta;
                }

                // Add squared error to total
                totalError += outputError * outputError;
            }

            // Store average training error for this epoch
            trainingLosses.add(totalError / inputs.length);

            // Print progress every 1000 epochs
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.15f%n", epoch, totalError / inputs.length);
            }
        }
    }

    // Run a forward pass for a new input and return denormalised prediction
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

    // Plot the training loss over epochs
    public static void plotErrorGraph(List<Double> trainingLosses) {
        XYSeries series = new XYSeries("Training Error");
        for (int i = 0; i < trainingLosses.size(); i++) {
            series.add(i, trainingLosses.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Training Error vs Epochs",
                "Epochs",
                "Error",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        NumberAxis yAxis = (NumberAxis) chart.getXYPlot().getRangeAxis();
        yAxis.setNumberFormatOverride(new java.text.DecimalFormat("0.000000"));

        JFrame frame = new JFrame("Training Error Graph");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    // Plot actual vs predicted values as a time series
    public static void plotActualVsPredicted(List<Date> dates, List<Double> actual, List<Double> predicted) {
        TimeSeries actualSeries = new TimeSeries("Actual Values");
        TimeSeries predictedSeries = new TimeSeries("Predicted Values");

        for (int i = 0; i < dates.size(); i++) {
            actualSeries.add(new Day(dates.get(i)), actual.get(i));
            predictedSeries.add(new Day(dates.get(i)), predicted.get(i));
        }

        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(actualSeries);
        dataset.addSeries(predictedSeries);

        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                "Actual vs Predicted Flow (Test Set)",
                "Date",
                "Flow Value",
                dataset,
                true, true, false);

        ((DateAxis) chart.getXYPlot().getDomainAxis())
            .setDateFormatOverride(new SimpleDateFormat("yyyy-MM-dd"));

        JFrame frame = new JFrame("Actual vs Predicted Flow");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }
}
