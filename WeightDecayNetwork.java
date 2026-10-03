package data;

import java.text.SimpleDateFormat;
import java.util.*;

import javax.swing.JFrame;

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

public class WeightDecayNetwork {
    private static final int INPUT_NODES = 2;
    private static final int HIDDEN_NODES = 5;
    private static final double LEARNING_RATE = 0.1;
    private static final double DECAY_RATE = 0.0005; // Beta value to control regularization

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double outputBias;

    private double outputMin;
    private double outputMax;

    public WeightDecayNetwork() {
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
        return sumSquares / 2.0; // Omega
    }

    public void train(double[][] trainingInputs, double[] expectedOutputs, int epochs,
                      List<Double> trainingLosses, List<Double> actualValues, List<Double> predictedValues) {

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double totalError = 0;

            for (int t = 0; t < trainingInputs.length; t++) {
                double[] inputs = trainingInputs[t];
                double expectedOutput = expectedOutputs[t];

                // Forward pass
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

                // Calculate loss with weight decay
                double error = expectedOutput - predictedOutput;
                double omega = computeWeightDecayPenalty();
                double beta = DECAY_RATE;
                double deltaOutput = (error + beta * omega) * sigmoidDerivative(predictedOutput);
                double lossWithDecay = error * error + beta * omega; // ℰ = E + βΩ
                totalError += lossWithDecay;

                // Backprop to hidden layer
                double[] deltaHidden = new double[HIDDEN_NODES];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    deltaHidden[j] = weightsHiddenOutput[j] * deltaOutput * sigmoidDerivative(hiddenOutputs[j]);
                }

                // Update weights hidden to output
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    weightsHiddenOutput[j] += LEARNING_RATE * deltaOutput * hiddenOutputs[j];
                }
                outputBias += LEARNING_RATE * deltaOutput;

                // Update weights input to hidden
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
                System.out.printf("Epoch %d - Error: %.10f\n", epoch, totalError / trainingInputs.length);
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
        double predictedOutput = sigmoid(outputSum);

        //System.out.println("Raw sigmoid output: " + predictedOutput);
        return denormalize(sigmoid(outputSum));
        
    }
    private double denormalize(double normalizedValue) {
      return (normalizedValue - 0.1) * (outputMax - outputMin) / 0.8 + outputMin;
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

} 

// Constructor for the Backpropogation_ReLU neural network class
public Backpropogation_ReLU() {
    Random rand = new Random(42); // Fixed seed for reproducibility

    // Initialise the weights and biases
    weightsInputHidden = new double[INPUT_NODES][HIDDEN_NODES]; // Input to hidden layer weights
    weightsHiddenOutput = new double[HIDDEN_NODES];             // Hidden to output layer weights
    hiddenBiases = new double[HIDDEN_NODES];                    // Bias values for hidden layer
    outputBias = rand.nextDouble() - 0.5;                       // Single bias for output neuron

    // Randomly assign small weights between input and hidden layer
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weightsInputHidden[i][j] = rand.nextDouble() - 0.5;
        }
    }

    // Randomly assign small weights and biases for hidden to output connections
    for (int j = 0; j < HIDDEN_NODES; j++) {
        weightsHiddenOutput[j] = rand.nextDouble() - 0.5;
        hiddenBiases[j] = rand.nextDouble() - 0.5;
    }
}

// Used to set the min and max of the target variable (needed for denormalisation later)
public void setOutputMinMax(double min, double max) {
    this.outputMin = min;
    this.outputMax = max;
}

// Standard sigmoid activation function
private double sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
}

// Derivative of the sigmoid activation (for use in backpropagation)
private double sigmoidDerivative(double x) {
    return x * (1 - x);
}

// ReLU (Rectified Linear Unit) activation function
private double relu(double x) {
    return Math.max(0, x);
}

// Derivative of ReLU function
private double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Helper class to store forward pass results
private static class ForwardResult {
    double[] hiddenOutputs; // ReLU activated outputs from hidden layer
    double[] hiddenSums;    // Raw sums (pre-activation) from hidden layer
    double finalOutput;     // Final predicted output after sigmoid
}

// Conducts a full forward pass given a set of input features
private ForwardResult forwardPassFull(double[] inputs) {
    double[] hiddenLayerOutputs = new double[HIDDEN_NODES];
    double[] hiddenSums = new double[HIDDEN_NODES];

    // Compute values for hidden layer
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double sum = hiddenBiases[j]; // Start with the bias
        for (int i = 0; i < INPUT_NODES; i++) {
            sum += inputs[i] * weightsInputHidden[i][j];
        }
        hiddenSums[j] = sum;                  // Store pre-activation sum
        hiddenLayerOutputs[j] = relu(sum);    // Apply ReLU activation
    }

    // Compute the final output layer activation
    double outputSum = outputBias;
    for (int j = 0; j < HIDDEN_NODES; j++) {
        outputSum += hiddenLayerOutputs[j] * weightsHiddenOutput[j];
    }

    // Store results in ForwardResult object
    ForwardResult result = new ForwardResult();
    result.hiddenOutputs = hiddenLayerOutputs;
    result.hiddenSums = hiddenSums;
    result.finalOutput = sigmoid(outputSum); // Output uses sigmoid
    return result;
}

