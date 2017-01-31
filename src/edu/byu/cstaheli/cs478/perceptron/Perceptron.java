package edu.byu.cstaheli.cs478.perceptron;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.Matrix;
import edu.byu.cstaheli.cs478.toolkit.SupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;

import java.util.Random;

/**
 * Created by cstaheli on 1/17/2017.
 */
public class Perceptron extends SupervisedLearner
{
    private Random random;
    private double learningRate;
    private double[] weights;
    private MLSystemManager manager;

    public Perceptron(Random rand, MLSystemManager manager)
    {
        super();
        setRandom(rand);
        setLearningRate(.1);
        this.setManager(manager);
    }

    @Override
    public void train(Matrix features, Matrix labels) throws Exception
    {

    }

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        initWeights(strategy.getTrainingFeatures());
        boolean keepTraining = true;
        int epochsWithoutSignificantImprovement = 0;
        double previousAccuracy = 0;
        double currentAccuracy = measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), null);
        double maxAccuracy = 0;
        getManager().completeEpoch(0, currentAccuracy);
        while (keepTraining)
        {
            previousAccuracy = currentAccuracy;
            int counter = 0;
            int correct = 0;
            Matrix features = strategy.getTrainingFeatures();
            Matrix labels = strategy.getTrainingLabels();
            for (int i = 0; i < features.rows(); ++i)
            {
                for (int j = 0; j < features.cols(); ++j)
                {
                    double input = features.get(i, j);
                    double expected = getExpected(labels, i);
                    double[] row = features.row(i);
                    double actual = getActivation(getWeights(), row);
                    double newWeight = calcNewWeight(getWeights()[j], getLearningRate(), expected, actual, input);
                    getWeights()[j] = newWeight;
                    ++counter;
                    if (expected == actual)
                    {
                        ++correct;
                    }
                }
            }
            currentAccuracy = (double) correct / counter;

            if (!isAccuracyChangeLargeEnough(maxAccuracy, currentAccuracy))
            {
                int EPOCHS_WITHOUT_SIGNIFICANT_IMPROVEMENT = 5;
                if (++epochsWithoutSignificantImprovement >= EPOCHS_WITHOUT_SIGNIFICANT_IMPROVEMENT)
                {
                    keepTraining = false;
                }
            }
            else
            {
                epochsWithoutSignificantImprovement = 0;
            }
            if (currentAccuracy > maxAccuracy)
            {
                maxAccuracy = currentAccuracy;
            }
            incrementTotalEpochs();
            getManager().completeEpoch(getTotalEpochs(), currentAccuracy);
        }
    }

    protected double getExpected(Matrix labels, int row)
    {
        return labels.get(row, 0);
    }

    protected double getActivation(double[] rowWeights, double[] row)
    {
        double sum = 0;
        for (int i = 0; i < row.length; ++i)
        {
            sum += (rowWeights[i] * row[i]);
        }
        return sum > 0 ? 1 : 0;
    }

    private boolean isAccuracyChangeLargeEnough(double previousAccuracy, double currentAccuracy)
    {
        return (currentAccuracy - previousAccuracy) >= .005;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        labels[0] = getActivation(getWeights(), features);
    }

    private double calcNewWeight(double oldWeight, double learningRate, double expected, double actual, double input)
    {
        return oldWeight - learningRate * (actual - expected) * input;
    }

    private void initWeights(Matrix features)
    {
        setWeights(new double[features.cols()]);
        for (int i = 0; i < features.cols(); ++i)
        {
            getWeights()[i] = getRandomWeight();
        }
    }

    private double getRandomWeight()
    {
        // Gives numbers between -.5 and .5
        return getRandom().nextDouble() - 0.5;
    }

    public double[] getWeights()
    {
        return weights;
    }

    public Random getRandom()
    {
        return random;
    }

    public void setRandom(Random random)
    {
        this.random = random;
    }

    public double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    public void setWeights(double[] weights)
    {
        this.weights = weights;
    }

    public MLSystemManager getManager()
    {
        return manager;
    }

    public void setManager(MLSystemManager manager)
    {
        this.manager = manager;
    }
}
