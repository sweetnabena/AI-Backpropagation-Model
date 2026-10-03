package data;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

public class DataAnalysis {
    public static void computeCorrelations(double[][] dataset) {
        if (dataset.length < 2) {
            System.out.println("Not enough data for correlation analysis.");
            return;
        }

        PearsonsCorrelation correlation = new PearsonsCorrelation();
        double[][] correlationMatrix = correlation.computeCorrelationMatrix(dataset).getData();

        for (int i = 0; i < correlationMatrix.length; i++) {
            for (int j = 0; j < correlationMatrix[i].length; j++) {
                System.out.print(String.format("%.2f ", correlationMatrix[i][j]));
            }
            System.out.println();
        }
    }
}

