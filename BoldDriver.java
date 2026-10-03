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



public class BoldDriver {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 30;
    private static final double MIN_LEARNING_RATE = 0.01;
    private static final double MAX_LEARNING_RATE = 0.5;

    private double learningRate = 0.1;

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;
    private double outputMin;
    private double outputMax;

    public BoldDriver() {
        Random rand = new Random(42);
        weightsInputHidden = new double[INPUT_NODES][HIDDEN_NODES];
        weightsHiddenOutput = new double[HIDDEN_NODES];
        hiddenBiases = new double[HIDDEN_NODES];
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

    private static class ForwardResult {
        double[] hiddenOutputs;
        double finalOutput;
    }

    private ForwardResult forwardPassFull(double[] inputs) {
        double[] hiddenLayerOutputs = new double[HIDDEN_NODES];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            double sum = hiddenBiases[j];
            for (int i = 0; i < INPUT_NODES; i++) {
                sum += inputs[i] * weightsInputHidden[i][j];
            }
            hiddenLayerOutputs[j] = sigmoid(sum);
        }

        double outputSum = outputBias;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            outputSum += hiddenLayerOutputs[j] * weightsHiddenOutput[j];
        }

        ForwardResult result = new ForwardResult();
        result.hiddenOutputs = hiddenLayerOutputs;
        result.finalOutput = sigmoid(outputSum);
        return result;
    }

    public void train(double[][] trainingInputs, double[] expectedOutputs, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues) {

        double previousError = Double.MAX_VALUE;

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;

            // Store copies of weights/biases to potentially revert
            double[][] prevWeightsIH = deepCopy(weightsInputHidden);
            double[] prevWeightsHO = Arrays.copyOf(weightsHiddenOutput, weightsHiddenOutput.length);
            double[] prevHiddenBiases = Arrays.copyOf(hiddenBiases, hiddenBiases.length);
            double prevOutputBias = outputBias;

            for (int t = 0; t < trainingInputs.length; t++) {
                double[] inputs = trainingInputs[t];
                double expectedOutput = expectedOutputs[t];

                ForwardResult result = forwardPassFull(inputs);
                double predictedOutput = result.finalOutput;

                if (epoch == epochs) {
                    actualValues.add(denormalize(expectedOutput));
                    predictedValues.add(denormalize(predictedOutput));
                }

                double error = expectedOutput - predictedOutput;
                totalError += error * error;

                double deltaOutput = error * sigmoidDerivative(predictedOutput);
                double[] deltaHidden = new double[HIDDEN_NODES];

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * sigmoidDerivative(result.hiddenOutputs[j]);
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    weightsHiddenOutput[j] += learningRate * deltaOutput * result.hiddenOutputs[j];
                }
                outputBias += learningRate * deltaOutput;

                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        weightsInputHidden[i][j] += learningRate * deltaHidden[j] * inputs[i];
                    }
                }
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    hiddenBiases[j] += learningRate * deltaHidden[j];
                }
            }

            double avgError = totalError / trainingInputs.length;
            trainingLosses.add(avgError);

            // Bold driver update every 1000 epochs
            if (epoch % 1000 == 0 && epoch > 0) {
                double errorChange = (avgError - previousError) / previousError;

                if (errorChange > 0.04) {  // Error increased significantly
                    weightsInputHidden = deepCopy(prevWeightsIH);
                    weightsHiddenOutput = Arrays.copyOf(prevWeightsHO, prevWeightsHO.length);
                    hiddenBiases = Arrays.copyOf(prevHiddenBiases, prevHiddenBiases.length);
                    outputBias = prevOutputBias;
                    learningRate = Math.max(MIN_LEARNING_RATE, learningRate * 0.7);
                } else if (errorChange < 0) {  // Error decreased
                    learningRate = Math.min(MAX_LEARNING_RATE, learningRate * 1.05);
                }
                previousError = avgError;
            }

            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.10f - Learning Rate: %.5f%n", epoch, avgError, learningRate);
            }
        }
    }

    private double[][] deepCopy(double[][] original) {
        double[][] copy = new double[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            copy[i] = Arrays.copyOf(original[i], original[i].length);
        }
        return copy;
    }

    public double predict(double[] input) {
        return denormalize(forwardPassFull(input).finalOutput);
    }

    private double denormalize(double normalizedValue) {
        return (normalizedValue - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
    }

    public static void plotErrorGraph(List<Double> trainingLosses) {
        XYSeries series = new XYSeries("Training Error");
        for (int i = 0; i < trainingLosses.size(); i++) {
            series.add(i, trainingLosses.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Training Error vs Epochs",
                "Epochs",
                "Error",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        NumberAxis yAxis = (NumberAxis) chart.getXYPlot().getRangeAxis();
        yAxis.setNumberFormatOverride(new java.text.DecimalFormat("0.000000"));

        JFrame frame = new JFrame("Training Error Graph");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

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
                true,
                true,
                false
        );

        ((DateAxis) chart.getXYPlot().getDomainAxis()).setDateFormatOverride(new SimpleDateFormat("yyyy-MM-dd"));

        JFrame frame = new JFrame("Actual vs Predicted Flow");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    // plotErrorGraph and plotActualVsPredicted remain unchanged
} 
