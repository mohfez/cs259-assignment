package prediction;

public abstract class PredictionModel
{
    public double[][] trainingData = new double[100][];
    public int[] trainingLabels = new int[100];
    public double[][] testingData = new double[100][];
    public int[] testingLabels = new int[100];

    public abstract double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice);
}