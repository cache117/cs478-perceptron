package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class StaticStrategy extends LearningStrategy
{
    private Matrix testData;

    public StaticStrategy(LearnerData learnerData) throws Exception
    {
        super(learnerData);
        Matrix testData = new Matrix();
        testData.loadArff(learnerData.getEvalParameter());
        if (learnerData.isNormalized())
            testData.normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
    }

    public Matrix getTestData()
    {
        return testData;
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
        return new Matrix(testData, 0, 0, testData.rows(), testData.cols() - 1);
    }

    @Override
    public Matrix getTestingLabels()
    {
        return new Matrix(testData, 0, testData.cols() - 1, testData.rows(), 1);
    }
}
