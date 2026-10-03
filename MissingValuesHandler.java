package data;

import java.util.List;

public class MissingValuesHandler {
    public static void handleMissingValues(List<List<String>> data) {
        if (data.isEmpty()) {
            System.out.println("Data list is empty. Skipping missing value handling.");
            return; // Return early if data is empty
        }
    
        int numCols = data.get(0).size(); // Assume all rows should have the same number of columns
        double[] columnSums = new double[numCols];
        int[] columnCounts = new int[numCols];
    
        // Compute column means
        for (List<String> row : data) {
            for (int i = 0; i < numCols; i++) {
                try {
                    double val = Double.parseDouble(row.get(i));
                    columnSums[i] += val;
                    columnCounts[i]++;
                } catch (NumberFormatException e) {
                    // Ignore non-numeric values
                }
            }
        }
    
        double[] columnMeans = new double[numCols];
        for (int i = 0; i < numCols; i++) {
            columnMeans[i] = columnCounts[i] > 0 ? columnSums[i] / columnCounts[i] : 0;
        }
    
        // Replace missing values with column means
        for (List<String> row : data) {
            for (int i = 0; i < numCols; i++) {
                if (row.get(i).isEmpty() || row.get(i).equalsIgnoreCase("NaN")) {
                    row.set(i, String.valueOf(columnMeans[i]));
                }
            }
        }
    }
    
}
