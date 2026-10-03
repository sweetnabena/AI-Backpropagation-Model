package data;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

public class DataSplitter {

    public static Map<String, List<List<String>>> splitData(List<List<String>> data) {
        System.out.println("Data size before splitting: " + data.size());
        Map<Integer, List<List<String>>> dataByYear = new HashMap<>();

        SimpleDateFormat dateFormat = new SimpleDateFormat("dd/MM/yyyy");
        for (List<String> row : data) {
            try {
                String dateStr = row.get(0);
                if (dateStr.equalsIgnoreCase("Date")) {
                    continue; // Skip leftover header row
                }
        
                Date date = dateFormat.parse(dateStr);
                @SuppressWarnings("deprecation")
                int year = date.getYear() + 1900;
        
                if (year >= 1993 && year <= 1996) {
                    dataByYear.computeIfAbsent(year, k -> new ArrayList<>()).add(row);
                }
            } catch (Exception e) {
                System.out.println("Invalid date format in first column: " + row.get(0));
            }
        }

        List<List<String>> trainSet = new ArrayList<>();
        List<List<String>> valSet = new ArrayList<>();
        List<List<String>> testSet = new ArrayList<>();

        if (dataByYear.containsKey(1993)) {
            trainSet.addAll(dataByYear.get(1993));
        }
        if (dataByYear.containsKey(1994)) {
            trainSet.addAll(dataByYear.get(1994));
        }
        if (dataByYear.containsKey(1995)) {
            valSet.addAll(dataByYear.get(1995));
        }
        if (dataByYear.containsKey(1996)) {
            testSet.addAll(dataByYear.get(1996));
        }

        System.out.println("Training Set: " + trainSet.size());
        System.out.println("Validation Set: " + valSet.size());
        System.out.println("Test Set: " + testSet.size());

        Map<String, List<List<String>>> result = new HashMap<>();
        result.put("train", trainSet);
        result.put("validation", valSet);
        result.put("test", testSet);

        return result;
    }
}
