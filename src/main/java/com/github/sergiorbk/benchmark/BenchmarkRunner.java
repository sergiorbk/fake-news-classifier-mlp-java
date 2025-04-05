package com.github.sergiorbk.benchmark;

import com.github.sergiorbk.data.DataLoader;
import com.github.sergiorbk.model.ParallelLogisticRegression;
import com.github.sergiorbk.model.SequentialLogisticRegression;

public class BenchmarkRunner {
    public static void runBenchmark(String dataPath) throws Exception {
        // load data
        long randomSeed = 2025;
        int numFeatures = 1000;
        int trainSize = 13713;
        int testSize = 4572;

        double[][] X_train = DataLoader.loadCSV(dataPath + "X_train.csv", trainSize, numFeatures);
        double[] y_train = DataLoader.loadLabels(dataPath + "y_train.csv", trainSize);
        double[][] X_test = DataLoader.loadCSV(dataPath + "X_test.csv", testSize, numFeatures);
        double[] y_test = DataLoader.loadLabels(dataPath + "y_test.csv", testSize);

        double learningRate = 0.05;
        int iterations = 1000;
        int batchSize = Math.max(256, X_train.length / Runtime.getRuntime().availableProcessors());
        double l2Lambda = 0.0001;

        // sequential learning
        SequentialLogisticRegression seqModel = new SequentialLogisticRegression(
                numFeatures, learningRate, iterations, batchSize, l2Lambda, randomSeed);

        long seqStartTime = System.currentTimeMillis();
        seqModel.train(X_train, y_train);
        long seqEndTime = System.currentTimeMillis();
        long seqDuration = seqEndTime - seqStartTime;

        double seqAccuracy = seqModel.evaluate(X_test, y_test);
        double seqAvgLoss = seqModel.computeAverageLoss(X_test, y_test);
        System.out.println("Sequential Test Accuracy: " + seqAccuracy + '\t' + "Avg Loss: " + seqAvgLoss);
        System.out.println("Sequential Time: " + seqDuration + "ms");
        System.out.println("==========================================================================================\n");

        // parallel learning
        ParallelLogisticRegression parModel = new ParallelLogisticRegression(
                numFeatures, learningRate, iterations, batchSize, l2Lambda, randomSeed);

        long parStartTime = System.currentTimeMillis();
        parModel.train(X_train, y_train);
        long parEndTime = System.currentTimeMillis();
        long parDuration = parEndTime - parStartTime;

        double parAccuracy = parModel.evaluate(X_test, y_test);
        double parAvgLoss = parModel.computeAverageLoss(X_test, y_test);
        System.out.println();
        System.out.println("Parallel Test Accuracy: " + parAccuracy + "\tAvg Loss: " + parAvgLoss + "\tParallel Time: " + parDuration + "ms");
        System.out.println();
        // speedup
        System.out.println("Sequential Time: " + seqDuration + "ms");
        System.out.println("Parallel Time: " + parDuration + "ms");
        double speedup = (double) seqDuration / parDuration;
        System.out.println("Speedup: " + String.format("%.2f", speedup) + "x");
    }
}
