package com.github.sergiorbk.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

public class ParallelLogisticRegression extends LogisticRegression {
    private final ForkJoinPool forkJoinPool;

    public ParallelLogisticRegression(int featureSize, double learningRate,
                                      int numIterations, int batchSize, double l2Lambda) {
        super(featureSize, learningRate, numIterations, batchSize, l2Lambda);
        this.forkJoinPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
    }

    @Override
    public void train(double[][] X, double[] y) {
        for (int iter = 0; iter < numIterations; iter++) {
            final double[] epochLoss = {0};

            List<Future<Double>> futures = new ArrayList<>();

            for (int i = 0; i < X.length; i += batchSize) {
                final int start = i;
                final int end = Math.min(i + batchSize, X.length);

                futures.add(forkJoinPool.submit(() -> {
                    double[][] batchX = Arrays.copyOfRange(X, start, end);
                    double[] batchY = Arrays.copyOfRange(y, start, end);
                    return processBatchSequential(batchX, batchY);
                }));
            }

            // collect results
            for (Future<Double> future : futures) {
                try {
                    epochLoss[0] += future.get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }

            double accuracy = evaluate(X, y);
            System.out.printf("Par Epoch %d, Loss: %.4f, Accuracy: %.4f\n",
                    iter + 1, epochLoss[0] / X.length, accuracy);
        }
    }

    private double processBatchSequential(double[][] batchX, double[] batchY) {
        double[] gradients = new double[weights.length];
        double biasGradient = 0;
        double batchLoss = 0;
        double[] currentWeights = this.weights; // local copy

        for (int i = 0; i < batchX.length; i++) {
            double prediction = predictProbability(batchX[i]);
            double error = prediction - batchY[i];

            for (int j = 0; j < weights.length; j++) {
                gradients[j] += error * batchX[i][j];
            }
            biasGradient += error;
            batchLoss += computeLoss(batchY[i], prediction);
        }

        // L2 regularization
        for (int j = 0; j < weights.length; j++) {
            gradients[j] += l2Lambda * currentWeights[j];
        }

        // atomic weights update
        synchronized (this) {
            for (int j = 0; j < weights.length; j++) {
                weights[j] -= learningRate * gradients[j] / batchX.length;
            }
            bias -= learningRate * biasGradient / batchX.length;
        }

        return batchLoss;
    }
}
