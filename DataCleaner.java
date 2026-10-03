package data;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class DataCleaner {
    public static List<List<String>> removeDuplicates(List<List<String>> data) {
        Set<String> uniqueRows = new HashSet<>();
        return data.stream()
                   .filter(row -> uniqueRows.add(String.join(",", row))) // Keep only unique rows
                   .collect(Collectors.toList());
    }
}
