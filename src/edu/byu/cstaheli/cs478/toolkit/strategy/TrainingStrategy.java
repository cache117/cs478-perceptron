package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class TrainingStrategy extends LearningStrategy
{
    public TrainingStrategy(LearnerData learnerData) throws Exception
    {
        super(learnerData);
    }

    @Override
    public Matrix getTrainingFeatures()
    {
        return new Matrix(getArffData(), 0, 0, getArffData().rows(), getArffData().cols() - 1);
    }

    @Override
    public Matrix getTrainingLabels()
    {
        return new Matrix(getArffData(), 0, getArffData().cols() - 1, getArffData().rows(), 1);
    }

    @Override
    public Matrix getTestingFeatures()
    {
        return getTrainingFeatures();
    }

    @Override
    public Matrix getTestingLabels()
    {
        return getTrainingLabels();
    }
}
