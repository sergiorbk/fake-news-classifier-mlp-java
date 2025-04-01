package com.github.sergiorbk.data;

import java.io.*;

public class DataLoader {
    public static double[][] loadCSV(String filePath, int rows, int cols) throws IOException {
        double[][] data = new double[rows][cols];
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        int rowIndex = 0;

        while ((line = br.readLine()) != null && rowIndex < rows) {
            String[] values = line.split(",");
            for (int col = 0; col < cols; col++) {
                data[rowIndex][col] = Double.parseDouble(values[col]);
            }
            rowIndex++;
        }
        br.close();
        return data;
    }

    public static double[] loadLabels(String filePath, int rows) throws IOException {
        double[] labels = new double[rows];
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        int rowIndex = 0;

        while ((line = br.readLine()) != null && rowIndex < rows) {
            labels[rowIndex] = Double.parseDouble(line.trim());
            rowIndex++;
        }
        br.close();
        return labels;
    }
}

