package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.Matrix;

/**
 * Created by cstaheli on 1/20/2017.
 */
public abstract class LearningStrategy
{
    private LearnerData learnerData;

    public LearningStrategy(LearnerData learnerData) throws Exception
    {
        this.learnerData = learnerData;
    }

    public abstract Matrix getTrainingFeatures();

    public abstract Matrix getTrainingLabels();

    public abstract Matrix getTestingFeatures();

    public abstract Matrix getTestingLabels();

    public Matrix getArffData()
    {
        return learnerData.getArffData();
    }
}
