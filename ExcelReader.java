package data;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

public class ExcelReader {
    public static List<List<String>> readExcelFile(String filePath) {
        List<List<String>> data = new ArrayList<>();
        boolean isFirstRow = true;

        try (FileInputStream fis = new FileInputStream(new File(filePath));
             Workbook workbook = new XSSFWorkbook(fis)) {

            SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy");
            Sheet sheet = workbook.getSheetAt(0); // First sheet

            
            for (Row row : sheet) {
                if (isFirstRow) {
                    isFirstRow = false;
                    continue;  // Skip the header row
                }

                List<String> rowData = new ArrayList<>();

                for (int i = 0; i < 9; i++) {
                    Cell cell = row.getCell(i, Row.MissingCellPolicy.CREATE_NULL_AS_BLANK);

                    if (i == 0 && cell.getCellType() == CellType.NUMERIC && DateUtil.isCellDateFormatted(cell)) {
                        rowData.add(formatter.format(cell.getDateCellValue()));  // Clean date
                    } else {
                        switch (cell.getCellType()) {
                            case STRING:
                                rowData.add(cell.getStringCellValue());
                                break;
                            case NUMERIC:
                                rowData.add(String.valueOf(cell.getNumericCellValue()));
                                break;
                            case BOOLEAN:
                                rowData.add(String.valueOf(cell.getBooleanCellValue()));
                                break;
                            case BLANK:
                                rowData.add("");
                                break;
                            default:
                                rowData.add("");
                                break;
                        }
                    }
                }

                if (rowData.size() == 9) {
                    data.add(rowData);
                } else {
                    System.out.println("Skipping row with invalid column count: " + rowData.size());
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Rows read: " + data.size());
        return data;
    }
}
