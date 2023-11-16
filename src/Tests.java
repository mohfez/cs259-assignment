//in progress creating template for the project (movie dataset reading, masking, KNN for K=1)
import java.io.*;
import java.util.*;

public class Tests
{
    static void Assert (boolean res) // We use this to test our results - don't delete or modify!
    {
        if (!res)
        {
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    static double dot(double[] U, double[] V)
    {
        double ans = 0;
        for (int i = 0; i < V.length; i++)
        {
            ans += U[i] * V[i];
        }

        return ans;
    }

    static int NumberOfFeatures = 2;
    static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice)
    {
        double[] feature = new double[NumberOfFeatures];
        feature[0] = id;  // We use the movie id as a numeric attribute.

        switch (genre)
        {
            // We also use represent each movie genre as an integer number:
            case "Action":  feature[1] = 0; break;
            case "Drama":   feature[1] = 1; break;
            case "Romance": feature[1] = 2; break;
            case "Sci-Fi": feature[1] = 3; break;
            case "Adventure": feature[1] = 4; break;
            case "Horror": feature[1] = 5; break;
            case "Mystery": feature[1] = 6; break;
            case "Thriller": feature[1] = 7; break;

        }

        // That is all. We don't use any other attributes for prediction.
        return feature;
    }

    // We are using the dot product to determine similarity:
    static double similarity(double[] u, double[] v)
    {
        return dot(u, v);
    }

    // KNN classifier for any K value?
    static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature, int k)
    {
        List<Map.Entry<Integer, Double>> similarities = new ArrayList<>();

        // store all indexes and similarities into a list
        for (int i = 0; i < trainingData.length; i++)
        {
            double currentSimilarity = similarity(testFeature, trainingData[i]);
            similarities.add(Map.entry(i, currentSimilarity));
        }

        // sort the list based on highest similarity to lowest
        similarities.sort((p1, p2) ->
        {
            if (p1.getValue() < p2.getValue()) return 1;
            else if (p1.getValue() > p2.getValue()) return -1;

            return 0;
        });

        // collect common labels using K
        HashMap<Integer, Integer> commonOccurrence = new HashMap<>();
        for (int i = 0; i < k; i++)
        {
            int label = trainingLabels[similarities.get(i).getKey()];
            if (commonOccurrence.containsKey(label))
            {
                commonOccurrence.put(label, commonOccurrence.get(label) + 1);
            }
            else commonOccurrence.put(label, 1);
        }

        // find the most common label
        return Collections.max(commonOccurrence.keySet(), (c1, c2) ->
        {
            if (commonOccurrence.get(c1) < commonOccurrence.get(c2)) return -1;
            else if (commonOccurrence.get(c1) > commonOccurrence.get(c2)) return 1;

            return 0;
        });
    }

    /**
     * @deprecated only supports k = 1
     */
    @Deprecated
    static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature)
    {
        int bestMatch = -1;
        double bestSimilarity = - Double.MAX_VALUE;  // We start with the worst similarity that we can get in Java.

        for (int i = 0; i < trainingData.length; i++)
        {
            double currentSimilarity = similarity(testFeature, trainingData[i]);
            if (currentSimilarity > bestSimilarity)
            {
                bestSimilarity = currentSimilarity;
                bestMatch = i;
            }
        }

        return trainingLabels[bestMatch];
    }


    static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels) throws IOException
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

                dataFeatures[idx] = toFeatureVector(id, genre, runtime, year, imdb, rt, budget, boxOffice);
                dataLabels[idx] = Integer.parseInt(values[11]); // Assuming the label is the last column and is numeric
                idx++;
            }
        }
    }

    public static void main(String[] args)
    {
        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try
        {
            // You may need to change the path:
            loadData("src/data/training-set.csv", trainingData, trainingLabels);
            loadData("src/data/testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e)
        {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        // Compute accuracy on the testing set
        int correctPredictions = 0;
        for (int i = 0; i < 100; i++)
        {
            if (knnClassify(trainingData, trainingLabels, testingData[i], 1) == testingLabels[i]) correctPredictions++;
        }

        double accuracy = (double) correctPredictions / testingData.length * 100;
        System.out.printf("A: %.2f%%\n", accuracy);
    }
}
