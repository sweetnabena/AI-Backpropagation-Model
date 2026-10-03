package data;

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

import javax.swing.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class Backpropagation_tanh {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 5;
    private static final double LEARNING_RATE = 0.1;

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;
    private double outputMin;
    private double outputMax;

    public Backpropagation_tanh() {
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

    private double tanh(double x) {
        return Math.tanh(x);
    }

    private double tanhDerivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private static class ForwardResult {
        double[] hiddenOutputs;
        double[] hiddenSums;
        double finalOutput;
    }

    private ForwardResult forwardPassFull(double[] inputs) {
        double[] hiddenLayerOutputs = new double[HIDDEN_NODES];
        double[] hiddenSums = new double[HIDDEN_NODES];

        for (int j = 0; j < HIDDEN_NODES; j++) {
            double sum = hiddenBiases[j];
            for (int i = 0; i < INPUT_NODES; i++) {
                sum += inputs[i] * weightsInputHidden[i][j];
            }
            hiddenSums[j] = sum;
            hiddenLayerOutputs[j] = tanh(sum);
        }

        double outputSum = outputBias;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            outputSum += hiddenLayerOutputs[j] * weightsHiddenOutput[j];
        }

        ForwardResult result = new ForwardResult();
        result.hiddenOutputs = hiddenLayerOutputs;
        result.hiddenSums = hiddenSums;
        result.finalOutput = sigmoid(outputSum); // output remains sigmoid
        return result;
    }

    public void train(double[][] trainingInputs, double[] expectedOutputs, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues) {
        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;

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
                    double tanhGrad = tanhDerivative(result.hiddenSums[j]);
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * tanhGrad;
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    weightsHiddenOutput[j] += LEARNING_RATE * deltaOutput * result.hiddenOutputs[j];
                }
                outputBias += LEARNING_RATE * deltaOutput;

                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        weightsInputHidden[i][j] += LEARNING_RATE * deltaHidden[j] * inputs[i];
                    }
                }
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    hiddenBiases[j] += LEARNING_RATE * deltaHidden[j];
                }
            }

            trainingLosses.add(totalError / trainingInputs.length);
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.15f%n", epoch, totalError / trainingInputs.length);
            }
        }
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
}
