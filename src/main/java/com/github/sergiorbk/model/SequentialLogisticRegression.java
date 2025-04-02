package com.github.sergiorbk.model;

import java.util.Arrays;

public class SequentialLogisticRegression extends LogisticRegression {
    public SequentialLogisticRegression(int featureSize, double learningRate,
                                        int numIterations, int batchSize, double l2Lambda) {
        super(featureSize, learningRate, numIterations, batchSize, l2Lambda);
    }

    @Override
    public void train(double[][] X, double[] y) {
        for (int iter = 0; iter < numIterations; iter++) {
            double totalLoss = 0;

            for (int i = 0; i < X.length; i += batchSize) {
                int end = Math.min(i + batchSize, X.length);
                double[][] batchX = Arrays.copyOfRange(X, i, end);
                double[] batchY = Arrays.copyOfRange(y, i, end);
                totalLoss += processBatch(batchX, batchY);
            }

            double accuracy = evaluate(X, y);
            System.out.printf("Seq Epoch %d, Loss: %.4f, Accuracy: %.4f\n",
                    iter + 1, totalLoss / X.length, accuracy);
        }
    }

    private double processBatch(double[][] batchX, double[] batchY) {
        double[] gradients = new double[weights.length];
        double biasGradient = 0;
        double batchLoss = 0;

        for (int i = 0; i < batchX.length; i++) {
            double prediction = predictProbability(batchX[i]);
            double error = prediction - batchY[i];

            for (int j = 0; j < weights.length; j++) {
                gradients[j] += error * batchX[i][j] + l2Lambda * weights[j];
            }
            biasGradient += error;
            batchLoss += computeLoss(batchY[i], prediction);
        }

        for (int j = 0; j < weights.length; j++) {
            weights[j] -= learningRate * gradients[j] / batchX.length;
        }
        bias -= learningRate * biasGradient / batchX.length;

        return batchLoss;
    }
}
