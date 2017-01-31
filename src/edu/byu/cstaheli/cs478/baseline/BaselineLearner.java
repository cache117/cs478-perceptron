package edu.byu.cstaheli.cs478.baseline;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.SupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

/**
 * For nominal labels, this model simply returns the majority class. For
 * continuous labels, it returns the mean value.
 * If the learning model you're using doesn't do as well as this one,
 * it's time to find a new learning model.
 */
public class BaselineLearner extends SupervisedLearner
{
    private double[] m_labels;

    @Override
    public void train(Matrix features, Matrix labels)
    {
        m_labels = new double[labels.cols()];
        for (int i = 0; i < labels.cols(); i++)
        {
            if (labels.valueCount(i) == 0)
                m_labels[i] = labels.columnMean(i); // continuous
            else
                m_labels[i] = labels.mostCommonValue(i); // nominal
        }
    }

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {

    }

    public void predict(double[] features, double[] labels) throws Exception
    {
        for (int i = 0; i < m_labels.length; i++)
            labels[i] = m_labels[i];
    }

}
