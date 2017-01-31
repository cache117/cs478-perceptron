package edu.byu.cstaheli.cs478.toolkit;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------


import edu.byu.cstaheli.cs478.baseline.BaselineLearner;
import edu.byu.cstaheli.cs478.perceptron.MultipleOutputPerceptron;
import edu.byu.cstaheli.cs478.perceptron.Perceptron;
import edu.byu.cstaheli.cs478.toolkit.strategy.*;

import java.io.*;
import java.util.Random;


public class MLSystemManager
{
    private Random random;
    private Matrix arffData;
    private SupervisedLearner learner;

    public MLSystemManager()
    {
        setRandom(new Random());
    }

    public static void main(String[] args) throws Exception
    {
        MLSystemManager ml = new MLSystemManager();
        ml.run(args);
    }

    /**
     * When you make a new learning algorithm, you should add a line for it to this method.
     */
    public SupervisedLearner getLearner(String model, Random rand) throws Exception
    {
        if (model.equals("baseline")) return new BaselineLearner();
        else if (model.equals("perceptron")) return new Perceptron(rand, this);
            // else if (model.equals("neuralnet")) return new NeuralNet(rand);
            // else if (model.equals("decisiontree")) return new DecisionTree();
            // else if (model.equals("knn")) return new InstanceBasedLearner();
        else throw new Exception("Unrecognized model: " + model);
    }

    public SupervisedLearner getLearner()
    {
        return learner;
    }

    public void run(String[] args) throws Exception
    {

        //args = new String[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

        //Parse the command line arguments
        ArgParser parser = new ArgParser(args);
        determineEvalMethod(parser);
    }

    public void setRandomSeed(long seed)
    {
        //Random rand = new Random(1234); // Use a seed for deterministic results (makes debugging easier)
        setRandom(new Random(seed));
    }

    private void determineEvalMethod(ArgParser parser) throws Exception
    {
        // Load the model
        if (parser.getWantedIndex().equals(""))
        {
            learner = getLearner(parser.getLearner(), getRandom());
        }
        else
        {
            learner = new MultipleOutputPerceptron(getRandom(), this, Double.parseDouble(parser.getWantedIndex()));
        }

        // Load the ARFF file
        arffData = new Matrix();
        arffData.loadArff(parser.getARFF());
        if (parser.isNormalized())
        {
            System.out.println("Using normalized data\n");
            arffData.normalize();
        }

        printStats(parser.getARFF(), parser.getLearner(), parser.getEvaluation(), arffData);

        LearnerData learnerData = new LearnerData(getRandom(), parser, arffData);
        switch (parser.getEvaluation())
        {
            case "training":
                calcTraining(learner, learnerData);
                break;
            case "static":
                calcStatic(learner, learnerData);
                break;
            case "random":
                calcRandom(learner, learnerData);
                break;
            case "cross":
                calcCrossValidation(learner, learnerData);
                break;
        }
    }

    private void calcTraining(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        System.out.println("Calculating accuracy on training set...");
        LearningStrategy strategy = new TrainingStrategy(learnerData);
//        Matrix features = new Matrix(learnerData.getArffData(), 0, 0, learnerData.getArffData().rows(), learnerData.getArffData().cols() - 1);
//        Matrix labels = new Matrix(learnerData.getArffData(), 0, learnerData.getArffData().cols() - 1, learnerData.getArffData().rows(), 1);
        Matrix confusion = new Matrix();
        double startTime = System.currentTimeMillis();
        learner.train(strategy);
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
        double accuracy = learner.measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), confusion);
        System.out.println("Training set accuracy: " + accuracy);
        if (learnerData.isVerbose())
        {
            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
            confusion.print();
            System.out.println("\n");
        }
    }

    private void calcStatic(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        LearningStrategy strategy = new StaticStrategy(learnerData);
//        Matrix testData = new Matrix();
//        testData.loadArff(learnerData.getEvalParameter());
//        if (learnerData.isNormalized())
//            testData.normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!

        System.out.println("Calculating accuracy on separate test set...");
        System.out.println("Test set name: " + learnerData.getEvalParameter());
        System.out.println("Number of test instances: " + ((StaticStrategy) strategy).getTestData().rows());
//        Matrix features = new Matrix(learnerData.getArffData(), 0, 0, learnerData.getArffData().rows(), learnerData.getArffData().cols() - 1);
//        Matrix labels = new Matrix(learnerData.getArffData(), 0, learnerData.getArffData().cols() - 1, learnerData.getArffData().rows(), 1);
        double startTime = System.currentTimeMillis();
        learner.train(strategy);
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
        double trainAccuracy = learner.measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), null);
        System.out.println("Training set accuracy: " + trainAccuracy);
        Matrix testFeatures = strategy.getTestingFeatures();
        Matrix testLabels = strategy.getTestingLabels();
//        Matrix testFeatures = new Matrix(testData, 0, 0, testData.rows(), testData.cols() - 1);
//        Matrix testLabels = new Matrix(testData, 0, testData.cols() - 1, testData.rows(), 1);
        Matrix confusion = new Matrix();
        double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
        System.out.println("Test set accuracy: " + testAccuracy);
        if (learnerData.isVerbose())
        {
            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
            confusion.print();
            System.out.println("\n");
        }
    }

    private void calcRandom(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        LearningStrategy strategy = new RandomStrategy(learnerData);
        System.out.println("Calculating accuracy on a random hold-out set...");
        double trainPercent = Double.parseDouble(learnerData.getEvalParameter());
        if (trainPercent < 0 || trainPercent > 1)
            throw new Exception("Percentage for random evaluation must be between 0 and 1");
        System.out.println("Percentage used for training: " + trainPercent);
        System.out.println("Percentage used for testing: " + (1 - trainPercent));
        learnerData.getArffData().shuffle(learnerData.getRandom());
//        int trainSize = (int) (trainPercent * data.rows());
//        Matrix trainFeatures = new Matrix(data, 0, 0, trainSize, data.cols() - 1);
//        Matrix trainLabels = new Matrix(data, 0, data.cols() - 1, trainSize, 1);
//        Matrix testFeatures = new Matrix(data, trainSize, 0, data.rows() - trainSize, data.cols() - 1);
//        Matrix testLabels = new Matrix(data, trainSize, data.cols() - 1, data.rows() - trainSize, 1);
        Matrix testFeatures = strategy.getTestingFeatures();
        Matrix testLabels = strategy.getTestingLabels();
        double startTime = System.currentTimeMillis();
        learner.train(strategy);
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
        double trainAccuracy = learner.measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), null);
        System.out.println("Training set accuracy: " + trainAccuracy);
        Matrix confusion = new Matrix();
        double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
        System.out.println("Test set accuracy: " + testAccuracy);
        if (learnerData.isVerbose())
        {
            System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
            confusion.print();
            System.out.println("\n");
        }
        writeAccuraciesAndFinalWeights(trainAccuracy, testAccuracy, ((Perceptron) learner).getWeights());
    }

    private void calcCrossValidation(SupervisedLearner learner, LearnerData learnerData) throws Exception
    {
        LearningStrategy strategy;
        System.out.println("Calculating accuracy using cross-validation...");
        int folds = Integer.parseInt(learnerData.getEvalParameter());
        if (folds <= 0)
            throw new Exception("Number of folds must be greater than 0");
        System.out.println("Number of folds: " + folds);
        int reps = 1;
        double sumAccuracy = 0.0;
        double elapsedTime = 0.0;
        for (int j = 0; j < reps; j++)
        {
            learnerData.getArffData().shuffle(learnerData.getRandom());
            for (int i = 0; i < folds; i++)
            {
                int begin = i * learnerData.getArffData().rows() / folds;
                int end = (i + 1) * learnerData.getArffData().rows() / folds;
                strategy = new CrossValidationStrategy(learnerData, begin, end);
//                Matrix trainFeatures = new Matrix(data, 0, 0, begin, data.cols() - 1);
//                Matrix trainLabels = new Matrix(data, 0, data.cols() - 1, begin, 1);
//                Matrix testFeatures = new Matrix(data, begin, 0, end - begin, data.cols() - 1);
//                Matrix testLabels = new Matrix(data, begin, data.cols() - 1, end - begin, 1);
                Matrix testFeatures = strategy.getTestingFeatures();
                Matrix testLabels = strategy.getTestingLabels();
//                trainFeatures.add(data, end, 0, data.rows() - end);
//                trainLabels.add(data, end, data.cols() - 1, data.rows() - end);
                double startTime = System.currentTimeMillis();
                learner.train(strategy);
                elapsedTime += System.currentTimeMillis() - startTime;
                double accuracy = learner.measureAccuracy(testFeatures, testLabels, null);
                sumAccuracy += accuracy;
                System.out.println("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
            }
        }
        elapsedTime /= (reps * folds);
        System.out.println("Average time to train (in seconds): " + elapsedTime / 1000.0);
        System.out.println("Mean accuracy=" + (sumAccuracy / (reps * folds)));
        System.out.println("Total number of epochs: " + learner.getTotalEpochs());
    }

    private void printStats(String fileName, String learnerName, String evalMethod, Matrix data)
    {
        // Print some stats
        System.out.println();
        System.out.println("Dataset name: " + fileName);
        System.out.println("Number of instances: " + data.rows());
        System.out.println("Number of attributes: " + data.cols());
        System.out.println("Learning algorithm: " + learnerName);
        System.out.println("Evaluation method: " + evalMethod);
        System.out.println();
    }

    public Random getRandom()
    {
        return random;
    }

    private void setRandom(Random random)
    {
        this.random = random;
    }

    public void completeEpoch(int epoch, double trainingAccuracy)
    {
        try (FileWriter writer = new FileWriter("datasets/accuracyVsEpochs.csv", true))
        {
            writer.append(String.format("%s, %s\n", epoch, trainingAccuracy));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public void writeAccuraciesAndFinalWeights(double trainingAccuracy, double testingAccuracy, double[] weights)
    {
        try (FileWriter writer = new FileWriter("datasets/accuracyVsEpochs.csv", true))
        {
            writer.append(String.format("Accuracy\n%s, %s\n", trainingAccuracy, testingAccuracy));
            writer.append(String.format("Final Weights\n%s", getArrayString(weights)));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    private String getArrayString(double[] array)
    {
        StringBuilder builder = new StringBuilder();
        for (double anArray : array)
        {
            builder.append(",").append(anArray);
        }
        return builder.toString();
    }
}
