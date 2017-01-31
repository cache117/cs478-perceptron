package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;

import java.util.Random;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class RandomStrategy extends LearningStrategy
{
    private double trainPercent;
    private Random rand;

    public RandomStrategy(LearnerData learnerData) throws Exception
    {
        super(learnerData);
        rand = learnerData.getRandom();
        trainPercent = Double.parseDouble(learnerData.getEvalParameter());
        if (trainPercent < 0 || trainPercent > 1)
            throw new Exception("Percentage for random evaluation must be between 0 and 1");
        System.out.println("Percentage used for training: " + trainPercent);
        System.out.println("Percentage used for testing: " + (1 - trainPercent));
        getArffData().shuffle(rand);
    }

    @Override
    public Matrix getTrainingFeatures()
    {
        getArffData().shuffle(rand);
        return new Matrix(getArffData(), 0, 0, getTrainSize(), getArffData().cols() - 1);
    }

    @Override
    public Matrix getTrainingLabels()
    {
        return new Matrix(getArffData(), 0, getArffData().cols() - 1, getTrainSize(), 1);
    }

    @Override
    public Matrix getTestingFeatures()
    {
        return new Matrix(getArffData(), getTrainSize(), 0, getArffData().rows() - getTrainSize(), getArffData().cols() - 1);
    }

    @Override
    public Matrix getTestingLabels()
    {
        return new Matrix(getArffData(), getTrainSize(), getArffData().cols() - 1, getArffData().rows() - getTrainSize(), 1);
    }

    private int getTrainSize()
    {
        return (int) (trainPercent * getArffData().rows());
    }
}
