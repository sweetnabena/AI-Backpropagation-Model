package data;

import java.text.SimpleDateFormat;
import java.util.*;

public class Main {
    public static void main(String[] args) {

        String filePath = "./data/Ouse93-96Student.xlsx";

        // Load and preprocess data
        List<List<String>> dataset = ExcelReader.readExcelFile(filePath);
        dataset = DataCleaner.removeDuplicates(dataset);
        MissingValuesHandler.handleMissingValues(dataset);

        // Split data
        Map<String, List<List<String>>> splitData = DataSplitter.splitData(dataset);
        List<List<String>> trainingSet = splitData.get("train");
        List<List<String>> validationSet = splitData.get("validation");
        List<List<String>> testSet = splitData.get("test");

        // Standardize data
        double[][] minMaxValues = computeMinMax(trainingSet);
        List<List<Double>> numericalTrainingData = standardiseData(trainingSet, minMaxValues[0], minMaxValues[1]);

        // Convert data to proper format
        double[][] trainingInputs = new double[numericalTrainingData.size()][2];
        double[] expectedOutputs = new double[numericalTrainingData.size()];

        for (int i = 0; i < numericalTrainingData.size(); i++) {
            List<Double> row = numericalTrainingData.get(i);
            if (row.size() > 3) {
                trainingInputs[i][0] = row.get(0); // Crakehill (normalized)
                trainingInputs[i][1] = row.get(2); // Westwick (normalized)
                expectedOutputs[i] = row.get(3);   // Skelton (normalized)
            } else {
                System.out.println("Skipping row with insufficient data: " + row);
            }
        }

        // Train MLP
        ReLUAnnealing mlp = new ReLUAnnealing();
        mlp.setOutputMinMax(minMaxValues[0][4], minMaxValues[1][4]); // Set before predictions

        List<Double> trainingLosses = new ArrayList<>();
        List<Double> actualValues = new ArrayList<>();
        List<Double> predictedValues = new ArrayList<>();
        List<Double> learningRateTrack = new ArrayList<>();
        mlp.train(trainingInputs, expectedOutputs, 27000, trainingLosses, actualValues, predictedValues, learningRateTrack);

        double minMSE = trainingLosses.stream().min(Double::compare).orElse(Double.NaN);

        // Predict on validation data
        List<Date> validationDates = new ArrayList<>();
        List<Double> validationActualValues = new ArrayList<>();
        List<Double> validationPredictedValues = new ArrayList<>();

        List<Date> testDates = new ArrayList<>();
        List<Double> testActualValues = new ArrayList<>();
        List<Double> testPredictedValues = new ArrayList<>();

        for (List<String> row : testSet) {
            try {
                Date date = new SimpleDateFormat("dd/MM/yyyy").parse(row.get(0));
                double[] input = {
                    0.8 * (Double.parseDouble(row.get(1)) - minMaxValues[0][1]) / (minMaxValues[1][1] - minMaxValues[0][1]) + 0.1,
                    0.8 * (Double.parseDouble(row.get(3)) - minMaxValues[0][3]) / (minMaxValues[1][3] - minMaxValues[0][3]) + 0.1
                };
                double actual = Double.parseDouble(row.get(4));

                double predicted = mlp.predict(input);

                //System.out.printf("Predicted: %.4f, Actual: %.4f%n", predicted, actual);

                testDates.add(date);
                testActualValues.add(actual);
                testPredictedValues.add(predicted);
            } catch (Exception e) {
                System.out.println("Skipping invalid row: " + row);
            }
        }

        System.out.println("First few rows of dataset:");
        for (int i = 0; i < 5; i++) {
            System.out.println(dataset.get(i));
        }

        // Show graphs
        ReLUAnnealing.plotErrorGraph(trainingLosses);
        ReLUAnnealing.plotScatterActualVsPredicted(testDates, testActualValues, testPredictedValues);
        ReLUAnnealing.plotLearningRate(learningRateTrack);

        // Raw test values (in real-world scale)
        double rawCrakehill = 0.5;
        double rawWestwick = 0.6;

        // Normalize to [0.1, 0.9] using training min/max
        double normalizedCrakehill = 0.8 * (rawCrakehill - minMaxValues[0][1]) / (minMaxValues[1][1] - minMaxValues[0][1]) + 0.1;
        double normalizedWestwick = 0.8 * (rawWestwick - minMaxValues[0][3]) / (minMaxValues[1][3] - minMaxValues[0][3]) + 0.1;

        double[] testSample = {normalizedCrakehill, normalizedWestwick};

        // Test prediction sample
        //double[] testSample = {0.5, 0.6}; // Normalized input example
        double prediction = mlp.predict(testSample);
        System.out.printf("Output Min: %.4f, Output Max: %.4f%n", minMaxValues[0][4], minMaxValues[1][4]);
        if (minMaxValues[0][4] == minMaxValues[1][4]) {
            System.err.println("⚠️ Warning: outputMin and outputMax are the same! Denormalization will divide by zero.");
        }
        
        System.out.printf("Predicted output for test sample: %.4f%n", prediction);
        System.out.printf("Min MSE: %.10f%n", minMSE);
    }

    public static double[][] computeMinMax(List<List<String>> trainingData) {
        int cols = trainingData.get(0).size();
        double[] minValues = new double[cols];
        double[] maxValues = new double[cols];
        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, -Double.MAX_VALUE);

        for (List<String> row : trainingData) {
            for (int j = 1; j < cols; j++) {
                try {
                    double value = Double.parseDouble(row.get(j));
                    if (value < minValues[j]) minValues[j] = value;
                    if (value > maxValues[j]) maxValues[j] = value;
                } catch (NumberFormatException e) {
                    // Ignore non-numeric values
                }
            }
        }
        return new double[][]{minValues, maxValues};
    }

    public static List<List<Double>> standardiseData(List<List<String>> data, double[] minValues, double[] maxValues) {
        List<List<Double>> numericalData = new ArrayList<>();
        int cols = data.get(0).size();

        for (List<String> row : data) {
            List<Double> standardizedRow = new ArrayList<>();
            for (int j = 1; j < cols; j++) {
                try {
                    double value = Double.parseDouble(row.get(j));
                    if (maxValues[j] != minValues[j]) {
                        standardizedRow.add(0.8 * (value - minValues[j]) / (maxValues[j] - minValues[j]) + 0.1);
                    } else {
                        standardizedRow.add(0.5);
                    }
                } catch (NumberFormatException e) {
                    standardizedRow.add(0.5);
                }
            }
            numericalData.add(standardizedRow);
        }
        return numericalData;
    }
}


