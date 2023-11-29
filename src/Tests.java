import prediction.*;

import java.io.*;

public class Tests
{
    static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels, PredictionModel model) throws IOException
    {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath)))
        {
            String line;
            int idx = 0;
            br.readLine(); // skip header line
            while ((line = br.readLine()) != null)
            {
                String[] values = line.split(",");
                // Assuming csv format: MovieID,Title,Genre,Runtime,Year,Lead Actor,Director,IMDB,RT(%),Budget,Box Office Revenue (in million $),Like it
                double id = Double.parseDouble(values[0]);
                String genre = values[2];
                double runtime = Double.parseDouble(values[3]);
                double year = Double.parseDouble(values[4]);
                double imdb = Double.parseDouble(values[7]);
                double rt = Double.parseDouble(values[8]);
                double budget = Double.parseDouble(values[9]);
                double boxOffice = Double.parseDouble(values[10]);

                dataFeatures[idx] = model.toFeatureVector(id, genre, runtime, year, imdb, rt, budget, boxOffice);
                dataLabels[idx] = Integer.parseInt(values[11]); // Assuming the label is the last column and is numeric
                idx++;
            }
        }
    }

    public static void main(String[] args)
    {
        // Compute accuracy on the testing set
        int knnCorrectPredictions = 0;
        int simpleProbCorrectPredictions = 0;
        int bayesCorrectPredictions = 0;
        int gaussianBayesCorrectPredictions = 0;

        KNN knn = new KNN();
        NaiveBayes naiveBayes = new NaiveBayes();
        GaussianNaiveBayes gaussianNaiveBayes = new GaussianNaiveBayes();
        SimpleProbabilities simpleProbabilities = new SimpleProbabilities();

        try
        {
            loadData("src/data/training-set.csv", knn.trainingData, knn.trainingLabels, knn);
            loadData("src/data/testing-set.csv", knn.testingData, knn.testingLabels, knn);

            loadData("src/data/training-set.csv", naiveBayes.trainingData, naiveBayes.trainingLabels, naiveBayes);
            loadData("src/data/testing-set.csv", naiveBayes.testingData, naiveBayes.testingLabels, naiveBayes);

            loadData("src/data/training-set.csv", gaussianNaiveBayes.trainingData, gaussianNaiveBayes.trainingLabels, gaussianNaiveBayes);
            loadData("src/data/testing-set.csv", gaussianNaiveBayes.testingData, gaussianNaiveBayes.testingLabels, gaussianNaiveBayes);

            loadData("src/data/training-set.csv", simpleProbabilities.trainingData, simpleProbabilities.trainingLabels, simpleProbabilities);
            loadData("src/data/testing-set.csv", simpleProbabilities.testingData, simpleProbabilities.testingLabels, simpleProbabilities);
        }
        catch (IOException e)
        {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        int dataLength = knn.testingData.length; // all lengths are the same anyway
        for (int i = 0; i < dataLength; i++)
        {
            // best k value is chosen using sqrt(n) and +/- 1 if it's even, so sqrt(100) = 10 - 1 = 9 or 10 + 1 = 11
            boolean knnClassify = knn.knnClassify(i, 11) == knn.testingLabels[i];
            boolean simpleProbabilitiesClassify = simpleProbabilities.simpleProbabilityModel(i) == simpleProbabilities.testingLabels[i];
            boolean naiveBayesClassify = naiveBayes.gaussianNaiveBayesClassify(i, new boolean[] { false }) == naiveBayes.testingLabels[i]; // we set all of continuousFeatures array to false so we can use the classical naive bayes calculations
            boolean gaussianNaiveBayesClassify = gaussianNaiveBayes.gaussianNaiveBayesClassify(i, new boolean[] { false, false, true, true, true }) == gaussianNaiveBayes.testingLabels[i]; // e.g. imdb is continuous, genre is not, hence it'll be { true, false }

            if (knnClassify) knnCorrectPredictions++;
            if (simpleProbabilitiesClassify) simpleProbCorrectPredictions++;
            if (naiveBayesClassify) bayesCorrectPredictions++;
            if (gaussianNaiveBayesClassify) gaussianBayesCorrectPredictions++;
        }

        double knnAccuracy = (double) knnCorrectPredictions / dataLength * 100;
        double simpleProbAccuracy = (double) simpleProbCorrectPredictions / dataLength * 100;
        double bayesAccuracy = (double) bayesCorrectPredictions / dataLength * 100;
        double gaussianBayesAccuracy = (double) gaussianBayesCorrectPredictions / dataLength * 100;
        System.out.printf("KNN Accuracy: %.2f%%\n", knnAccuracy);
        System.out.printf("Simple Prediction Model Accuracy: %.2f%%\n", simpleProbAccuracy);
        System.out.printf("Naive Bayes Accuracy: %.2f%%\n", bayesAccuracy);
        System.out.printf("Gaussian Naive Bayes Accuracy: %.2f%%\n", gaussianBayesAccuracy);
    }
}
