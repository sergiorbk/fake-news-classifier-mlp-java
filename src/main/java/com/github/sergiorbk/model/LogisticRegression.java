package com.github.sergiorbk.model;

import java.util.Random;

public abstract class LogisticRegression {
    protected double[] weights;
    protected double bias;
    protected final double learningRate;
    protected final int numIterations;
    protected final int batchSize;
    protected final double l2Lambda;
    protected final Random random;

    public LogisticRegression(int featureSize, double learningRate,
                              int numIterations, int batchSize, double l2Lambda, long seed) {
        this.weights = new double[featureSize];
        this.bias = 0;
        this.learningRate = learningRate;
        this.numIterations = numIterations;
        this.batchSize = batchSize;
        this.l2Lambda = l2Lambda;
        this.random = new Random(seed);
        initializeWeights();
    }

    protected void initializeWeights() {
        double stddev = 1.0 / Math.sqrt(weights.length);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = stddev * random.nextGaussian();
        }
    }

    public abstract void train(double[][] X, double[] y);

    public double evaluate(double[][] X, double[] y) {
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            if ((predictProbability(X[i]) > 0.5 ? 1 : 0) == y[i])
                correct++;
        }
        return (double) correct / X.length;
    }

    protected double predictProbability(double[] features) {
        double z = bias;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * features[i];
        }
        return sigmoid(z);
    }

    protected double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    protected double computeLoss(double trueLabel, double prediction) {
        double epsilon = 1e-15;
        prediction = Math.max(epsilon, Math.min(1 - epsilon, prediction));
        return -(trueLabel * Math.log(prediction) + (1 - trueLabel) * Math.log(1 - prediction));
    }

    public double computeAverageLoss(double[][] X, double[] y) {
        double totalLoss = 0;
        for (int i = 0; i < X.length; i++) {
            double prediction = predictProbability(X[i]);
            totalLoss += computeLoss(y[i], prediction);
        }
        return totalLoss / X.length;
    }

    public long getCurrentSeed() {
        return random.hashCode();
    }
}