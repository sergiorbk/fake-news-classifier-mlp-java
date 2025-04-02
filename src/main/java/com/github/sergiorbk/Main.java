package com.github.sergiorbk;

import com.github.sergiorbk.benchmark.BenchmarkRunner;

public class Main {
    public static void main(String[] args) {
        try {
            String dataPath = "C:\\Users\\serge\\.spyder-py3\\";
            BenchmarkRunner.runBenchmark(dataPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
