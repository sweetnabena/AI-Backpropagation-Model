package data;

import java.awt.Color;
import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Day;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;


public class ReLUAnnealing {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 18;
    private static final double p = 0.01;
    private static final double q = 0.5;
    private static final double r = 27000.0;

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;

    private double outputMin;
    private double outputMax;

    public void setOutputMinMax(double min, double max) {
        this.outputMin = min;
        this.outputMax = max;
    }

    private double denormalize(double normVal) {
        return (normVal - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
    }

    public ReLUAnnealing() {
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

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    private double computeAnnealedLearningRate(int epoch) {
        return p + (q - p) * (1 - 1.0 / (1 + Math.exp((10 - 20.0 * epoch / r))));
    }

    private static class ForwardResult {
        double[] hiddenOutputs;
        double[] hiddenSums;
        double finalOutput;
    }

    private ForwardResult forwardPass(double[] input) {
        ForwardResult result = new ForwardResult();
        result.hiddenOutputs = new double[HIDDEN_NODES];
        result.hiddenSums = new double[HIDDEN_NODES];

        for (int j = 0; j < HIDDEN_NODES; j++) {
            double sum = hiddenBiases[j];
            for (int i = 0; i < INPUT_NODES; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            result.hiddenSums[j] = sum;
            result.hiddenOutputs[j] = relu(sum);
        }

        double outputSum = outputBias;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            outputSum += result.hiddenOutputs[j] * weightsHiddenOutput[j];
        }
        result.finalOutput = sigmoid(outputSum);

        return result;
    }

    public void train(double[][] inputs, double[] targets, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues,
                      List<Double> learningRateTrack) {

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;
            double learningRate = computeAnnealedLearningRate(epoch);
            learningRateTrack.add(learningRate);

            for (int t = 0; t < inputs.length; t++) {
                double[] input = inputs[t];
                double expected = targets[t];

                ForwardResult result = forwardPass(input);
                double predicted = result.finalOutput;

                if (epoch == epochs) {
                    actualValues.add(denormalize(expected));
                    predictedValues.add(denormalize(predicted));
                }

                double error = expected - predicted;
                double deltaOutput = error * sigmoidDerivative(predicted);

                double[] deltaHidden = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * reluDerivative(result.hiddenSums[j]);
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    weightsHiddenOutput[j] += learningRate * deltaOutput * result.hiddenOutputs[j];
                }
                outputBias += learningRate * deltaOutput;

                for (int i = 0; i < INPUT_NODES; i++) {
                    for (int j = 0; j < HIDDEN_NODES; j++) {
                        weightsInputHidden[i][j] += learningRate * deltaHidden[j] * input[i];
                    }
                }
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    hiddenBiases[j] += learningRate * deltaHidden[j];
                }

                totalError += error * error;
            }

            trainingLosses.add(totalError / inputs.length);
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.10f - Learning Rate: %.5f%n",
                        epoch, totalError / inputs.length, learningRate);
            }
        }
    }

    public double predict(double[] input) {
        return denormalize(forwardPass(input).finalOutput);
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

         // Set white background
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);

        NumberAxis yAxis = (NumberAxis) chart.getXYPlot().getRangeAxis();
        yAxis.setNumberFormatOverride(new java.text.DecimalFormat("0.000000"));

        JFrame frame = new JFrame("Training Error Graph");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    public static void plotLearningRate(List<Double> learningRates) {
        XYSeries series = new XYSeries("Learning Rate");
        for (int i = 0; i < learningRates.size(); i++) {
            series.add(i, learningRates.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Learning Rate vs Epochs",
                "Epochs",
                "Learning Rate",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

         // Set white background
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);

        JFrame frame = new JFrame("Learning Rate Schedule");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    public static void plotScatterActualVsPredicted(List<Date> dates, List<Double> actual, List<Double> predicted) {
    XYSeries series = new XYSeries("Actual vs Predicted");

    for (int i = 0; i < dates.size(); i++) {
        series.add(actual.get(i), predicted.get(i)); // Scatter: x = actual, y = predicted
    }

    XYSeriesCollection dataset = new XYSeriesCollection();
    dataset.addSeries(series);

    JFreeChart chart = ChartFactory.createScatterPlot(
            "Actual vs Predicted (Test Set)",
            "Actual Flow Value",
            "Predicted Flow Value",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
    );

    // Set white background
    XYPlot plot = chart.getXYPlot();
    plot.setBackgroundPaint(Color.WHITE);

    JFrame frame = new JFrame("Scatter Plot - Test Results");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.add(new ChartPanel(chart));
    frame.pack();
    frame.setVisible(true);
}

}
