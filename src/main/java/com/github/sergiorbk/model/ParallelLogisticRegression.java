package com.github.sergiorbk.model;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

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

            forkJoinPool.submit(() ->
                    IntStream.range(0, (X.length + batchSize - 1) / batchSize)
                            .parallel()
                            .forEach(batchIdx -> {
                                int start = batchIdx * batchSize;
                                int end = Math.min(start + batchSize, X.length);
                                double[][] batchX = Arrays.copyOfRange(X, start, end);
                                double[] batchY = Arrays.copyOfRange(y, start, end);

                                double batchLoss = processBatch(batchX, batchY);
                                synchronized (epochLoss) {
                                    epochLoss[0] += batchLoss;
                                }
                            })
            ).join();

            double accuracy = evaluate(X, y);
            System.out.printf("Par Epoch %d, Loss: %.4f, Accuracy: %.4f\n",
                    iter + 1, epochLoss[0] / X.length, accuracy);
        }
    }

    private double processBatch(double[][] batchX, double[] batchY) {
        double[] localGradients = new double[weights.length];
        double localBiasGradient = 0;
        double localLoss = 0;
        double[] currentWeights = this.weights;

        for (int i = 0; i < batchX.length; i++) {
            double prediction = predictProbability(batchX[i]);
            double error = prediction - batchY[i];

            for (int j = 0; j < weights.length; j++) {
                localGradients[j] += error * batchX[i][j];
            }
            localBiasGradient += error;
            localLoss += computeLoss(batchY[i], prediction);
        }

        for (int j = 0; j < weights.length; j++) {
            localGradients[j] += l2Lambda * currentWeights[j];
        }

        synchronized (this) {
            for (int j = 0; j < weights.length; j++) {
                weights[j] -= learningRate * localGradients[j] / batchX.length;
            }
            bias -= learningRate * localBiasGradient / batchX.length;
        }

        return localLoss;
    }
}
