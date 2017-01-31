package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class CrossValidationStrategy extends LearningStrategy
{
    private int begin;
    private int end;
    private Matrix trainingFeatures;
    private Matrix trainingLabels;

    public CrossValidationStrategy(LearnerData learnerData) throws Exception
    {
        this(learnerData, 0, 0);
    }

    public CrossValidationStrategy(LearnerData learnerData, int begin, int end) throws Exception
    {
        super(learnerData);
        this.begin = begin;
        this.end = end;
        this.trainingFeatures = new Matrix(getArffData(), 0, 0, begin, getArffData().cols() - 1);
        this.trainingLabels = new Matrix(getArffData(), 0, getArffData().cols() - 1, begin, 1);
        trainingLabels.add(getArffData(), end, getArffData().cols() - 1, getArffData().rows() - end);
        trainingFeatures.add(getArffData(), end, 0, getArffData().rows() - end);

    }

    @Override
    public Matrix getTrainingFeatures()
    {
        return trainingFeatures;
    }

    @Override
    public Matrix getTrainingLabels()
    {
        return trainingLabels;
    }

    @Override
    public Matrix getTestingFeatures()
    {
        return new Matrix(getArffData(), begin, 0, end - begin, getArffData().cols() - 1);
    }

    @Override
    public Matrix getTestingLabels()
    {
        return new Matrix(getArffData(), begin, getArffData().cols() - 1, end - begin, 1);
    }
}
