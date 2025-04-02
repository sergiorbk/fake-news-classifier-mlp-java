package com.github.sergiorbk.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class DataLoader {
    public static double[][] loadCSV(String filePath, int numRows, int numFeatures) throws IOException {
        double[][] data = new double[numRows][numFeatures];
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null && row < numRows) {
                String[] values = line.split(",");
                for (int col = 0; col < numFeatures; col++) {
                    data[row][col] = Double.parseDouble(values[col]);
                }
                row++;
            }
        }
        return data;
    }

    public static double[] loadLabels(String filePath, int numRows) throws IOException {
        double[] labels = new double[numRows];
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null && row < numRows) {
                labels[row] = Double.parseDouble(line);
                row++;
            }
        }
        return labels;
    }
}