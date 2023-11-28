package prediction;

import java.util.*;

public class KNN extends PredictionModel
{
    @Override
    public double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice)
    {
        double[] feature = new double[1]; // n features

        // this gives 67% in k=1, 64% in k=11
        /*switch (genre)
        {
            // Use one-hot encoding for genre
            case "Action":  feature[0] = 1; break;
            case "Drama":   feature[1] = 1; break;
            case "Romance": feature[2] = 1; break;
            case "Sci-Fi": feature[3] = 1; break;
            case "Adventure": feature[4] = 1; break;
            case "Horror": feature[5] = 1; break;
            case "Mystery": feature[6] = 1; break;
            case "Thriller": feature[7] = 1; break;
        }
        feature[8] = Math.log10(runtime); // log transformation, using base 10 as constant*/

        switch (genre)
        {
            // Use sorted like/dislike ratio genre list (found in excel)
            case "Thriller": feature[0] = 1; break;
            case "Action": feature[0] = 2; break;
            case "Sci-Fi": feature[0] = 3; break;
            case "Adventure": feature[0] = 4; break;
            case "Romance": feature[0] = 5; break;
            case "Horror": feature[0] = 6; break;
            case "Drama":   feature[0] = 7; break;
            case "Mystery":  feature[0] = 8; break;
        }
        feature[0] -= 4.35; // average transformation

        return feature;
    }

    private double dot(double[] U, double[] V)
    {
        double ans = 0;
        for (int i = 0; i < V.length; i++)
        {
            ans += U[i] * V[i];
        }

        return ans;
    }

    // We are using the dot product to determine similarity:
    private double similarity(double[] u, double[] v)
    {
        return dot(u, v);
    }

    // KNN classifier for any K value
    public int knnClassify(int testingDataIndex, int k)
    {
        // list of all similarities with their respective label index
        List<Map.Entry<Integer, Double>> similarities = new ArrayList<>();

        // store all indexes and similarities into a list
        for (int i = 0; i < trainingData.length; i++)
        {
            double currentSimilarity = similarity(testingData[testingDataIndex], trainingData[i]);
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
            commonOccurrence.put(label, commonOccurrence.getOrDefault(label, 0) + 1);
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
    public int knnClassify(int testingDataIndex)
    {
        int bestMatch = -1;
        double bestSimilarity = - Double.MAX_VALUE;  // We start with the worst similarity that we can get in Java.

        for (int i = 0; i < trainingData.length; i++)
        {
            double currentSimilarity = similarity(testingData[testingDataIndex], trainingData[i]);
            if (currentSimilarity > bestSimilarity)
            {
                bestSimilarity = currentSimilarity;
                bestMatch = i;
            }
        }

        return trainingLabels[bestMatch];
    }
}
