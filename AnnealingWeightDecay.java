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

public class AnnealingWeightDecay {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 14;
    private static final double p = 0.01;
    private static final double q = 0.5;
    private static final double r = 30000.0;
    private static final double DECAY_RATE = 0.0005;

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;
    private double outputMin;
    private double outputMax;

    public AnnealingWeightDecay() {
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

    private double computeWeightDecayPenalty() {
        double sumSquares = 0.0;
        for (double[] row : weightsInputHidden) {
            for (double w : row) {
                sumSquares += w * w;
            }
        }
        for (double w : weightsHiddenOutput) {
            sumSquares += w * w;
        }
        return sumSquares / 2.0;
    }

    private double computeAnnealedLearningRate(int x) {
        return p + (q - p) * (1 - 1.0 / (1 + Math.exp((10 - 20.0 * x / r))));
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
                double expectedOutput = expectedOutputs[t];

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
                double predictedOutput = sigmoid(outputSum);

                if (epoch == epochs) {
                    actualValues.add(denormalize(expectedOutput));
                    predictedValues.add(denormalize(predictedOutput));
                }

                double error = expectedOutput - predictedOutput;
                double omega = computeWeightDecayPenalty();
                double deltaOutput = (error + DECAY_RATE * omega) * sigmoidDerivative(predictedOutput);
                totalError += error * error + DECAY_RATE * omega;

                double[] deltaHidden = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * sigmoidDerivative(hiddenOutputs[j]);
                }

                for (int j = 0; j < HIDDEN_NODES; j++) {
                    weightsHiddenOutput[j] += learningRate * deltaOutput * hiddenOutputs[j];
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

            trainingLosses.add(totalError / trainingInputs.length);
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.10f - Learning Rate: %.5f%n",
                        epoch, totalError / trainingInputs.length, learningRate);
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

    private double denormalize(double normalizedValue) {
        return (normalizedValue - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
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

        JFrame frame = new JFrame("Learning Rate Schedule");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
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
