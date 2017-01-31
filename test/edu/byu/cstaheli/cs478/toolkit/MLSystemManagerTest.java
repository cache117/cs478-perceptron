package edu.byu.cstaheli.cs478.toolkit;

import edu.byu.cstaheli.cs478.perceptron.Perceptron;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Created by cstaheli on 1/19/2017.
 */
class MLSystemManagerTest
{
    @Test
    void testBaseline() throws Exception
    {
        String[] args = "-L baseline -A datasets/iris.arff -E training".split(" ");
        MLSystemManager ml = new MLSystemManager();
        ml.setRandomSeed(1234);
        ml.run(args);
    }

    @Test
    void testPerceptron() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        //manager.setRandomSeed(1234);
//        System.out.println("Training");
//        args = "-L perceptron -A datasets/voting.arff -E training".split(" ");
//        manager.run(args);
//        System.out.println("Cross Fold Validation");
//        args = "-L perceptron -A datasets/voting.arff -E cross 25 -V".split(" ");
//        manager.run(args);
//        System.out.println("Training");
//        args = "-L perceptron -A datasets/first.arff -E training -V".split(" ");
//        manager.run(args);
//        System.out.println("Random - 95%");
//        args = "-L perceptron -A datasets/voting.arff -E random .95 -V".split(" ");
//        manager.run(args);

//        File csvFile = new File("datasets/accuracyVsEpochs.csv");
//        assertTrue(csvFile.delete());
////        args = "-L perceptron -A datasets/first.arff -E training".split(" ");
//        args = "-L perceptron -A datasets/second.arff -E training".split(" ");
//        manager.run(args);
//        System.out.println("Random - 70/30%");
//        args = "-L perceptron -A datasets/voting.arff -E random .7 -V".split(" ");
//        manager.run(args);
//        try (FileWriter writer = new FileWriter("datasets/accuracyVsEpochs.csv", true))
//        {
////            writer.append(String.format("Final Weights: %s\n", getArrayString(((Perceptron) manager.getLearner()).getWeights())));
//            System.out.println(String.format("Final Weights: %s\n", getArrayString(((Perceptron) manager.getLearner()).getWeights())));
//        }
//        catch (IOException e)
//        {
//            e.printStackTrace();
//        }
        args = "-L perceptron -A datasets/iris.arff -W 0 -E training".split(" ");
        manager.run(args);
        args = "-L perceptron -A datasets/iris.arff -W 1 -E training".split(" ");
        manager.run(args);
        args = "-L perceptron -A datasets/iris.arff -W 2 -E training".split(" ");
        manager.run(args);
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