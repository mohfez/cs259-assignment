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
        feature[0] = imdb; // TODO: edit feature vectors later

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

        return feature;
    }

    // We are using the dot product to determine similarity:
    static double similarity(double[] u, double[] v)
    {
        return dot(u, v);
    }

    // KNN classifier for any K value
    static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature, int k)
    {
        // list of all similarities with their respective label index
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

    /**
     * @deprecated does not support continuous features, use gaussianNaiveBayesClassify instead. Just set all the continuousFeatures array elements to false if you want the classical naive bayes.
     */
    @Deprecated
    static int naiveBayesClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature)
    {
        // total likes and dislikes
        int totalLikeCount = 0;
        int totalDislikeCount = 0;

        // maps of feature i : (feature val, (dis)like count)
        Map<Integer, Map<Double, Integer>> likeCount = new HashMap<>();
        Map<Integer, Map<Double, Integer>> dislikeCount = new HashMap<>();

        // train the classifier
        for (int i = 0; i < trainingLabels.length; i++)
        {
            // increment (dis)like counts
            if (trainingLabels[i] == 1) totalLikeCount++;
            else if (trainingLabels[i] == 0) totalDislikeCount++;

            for (int k = 0; k < trainingData[i].length; k++)
            {
                if (trainingLabels[i] == 1)
                {
                    likeCount.put(k, likeCount.getOrDefault(k, new HashMap<>())); // make a new hashmap if not already exists
                    Map<Double, Integer> featureMap = likeCount.get(k); // get the feature map (feature val, like occurrences)
                    featureMap.put(trainingData[i][k], featureMap.getOrDefault(trainingData[i][k], 0) + 1); // use feature value as key to add 1 like to
                }
                else if (trainingLabels[i] == 0)
                {
                    dislikeCount.put(k, dislikeCount.getOrDefault(k, new HashMap<>())); // make a new hashmap if not already exists
                    Map<Double, Integer> featureMap = dislikeCount.get(k); // get the feature map (feature val, dislike occurrences)
                    featureMap.put(trainingData[i][k], featureMap.getOrDefault(trainingData[i][k], 0) + 1); // use feature value as key to add 1 dislike to
                }
            }
        }

        // calculate probabilities
        double priorLikeProbability = (double) totalLikeCount / trainingData.length;
        double priorDislikeProbability = (double) totalDislikeCount / trainingData.length;

        double likeLikelihood = 1;
        double dislikeLikelihood = 1;

        for (int i = 0; i < testFeature.length; i++)
        {
            // amount of likes so far based on current feature value
            int countingLikes = likeCount.get(i).getOrDefault(testFeature[i], 0);
            // amount of dislikes so far based on current feature value
            int countingDislikes = dislikeCount.get(i).getOrDefault(testFeature[i], 0);

            likeLikelihood *= (double) countingLikes / trainingData.length;
            dislikeLikelihood *= (double) countingDislikes / trainingData.length;
        }

        // posterior probability = (likelihood * prior probability) / probability of evidence
        // based on p(a|b) = (p(b|a) * p(a)) / (p(b|a) * p(a) + p(b|~a) * p(~a))
        // or p(a|b,c..) = (p(b|a) * p(c|a) * p(a)) / (p(b|a) * p(c|a) * p(a) + p(b|~a) * p(c|~a) * p(~a))
        double evidence = (likeLikelihood * priorLikeProbability) + (dislikeLikelihood * priorDislikeProbability);
        double likePosteriorProbability = (likeLikelihood * priorLikeProbability) / evidence;
        double dislikePosteriorProbability = (dislikeLikelihood * priorDislikeProbability) / evidence;

        return likePosteriorProbability > dislikePosteriorProbability ? 1 : 0;
    }

    /*
        Uses nested hashmap to store the feature values alongside their like/dislike counts.
        feature i is the index of the features array that will be used to compare testing feature values with training feature values, e.g. feature[0] is year and feature[1] is genre (0 and 1)
        At the end, it'll find the occurrences for each feature type & value and use them for likelihood calculations

        Since the dataset has many continuous variables, gaussian naive bayes might be more appropriate instead of the classical
        runtime, year, imdb, rt, budget, box office all look like good features that are continuous

        Note: set all continuousFeatures array to false if you want to use the classical naive bayes model
     */
    static int gaussianNaiveBayesClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature, boolean[] continuousFeatures)
    {
        // total likes and dislikes
        int totalLikeCount = 0;
        int totalDislikeCount = 0;

        // maps of feature i : (feature val, (dis)like count)
        Map<Integer, Map<Double, Integer>> likeCount = new HashMap<>();
        Map<Integer, Map<Double, Integer>> dislikeCount = new HashMap<>();

        // feature i, double[4] [mean for like it, mean for dislike it, std. dev for like it, std. dev for dislike it]
        Map<Integer, double[]> meanAndStdDevTable = new HashMap<>();

        // train the classifier
        for (int i = 0; i < trainingLabels.length; i++)
        {
            // increment (dis)like counts
            if (trainingLabels[i] == 1) totalLikeCount++;
            else if (trainingLabels[i] == 0) totalDislikeCount++;

            for (int k = 0; k < trainingData[i].length; k++)
            {
                if (trainingLabels[i] == 1)
                {
                    if (!continuousFeatures[k])
                    {
                        likeCount.put(k, likeCount.getOrDefault(k, new HashMap<>())); // make a new hashmap if not already exists
                        Map<Double, Integer> featureMap = likeCount.get(k); // get the feature map (feature val, like occurrences)
                        featureMap.put(trainingData[i][k], featureMap.getOrDefault(trainingData[i][k], 0) + 1); // use feature value as key to add 1 like to
                    }
                }
                else if (trainingLabels[i] == 0)
                {
                    if (!continuousFeatures[k])
                    {
                        dislikeCount.put(k, dislikeCount.getOrDefault(k, new HashMap<>())); // make a new hashmap if not already exists
                        Map<Double, Integer> featureMap = dislikeCount.get(k); // get the feature map (feature val, dislike occurrences)
                        featureMap.put(trainingData[i][k], featureMap.getOrDefault(trainingData[i][k], 0) + 1); // use feature value as key to add 1 dislike to
                    }
                }

                // process for calculating means
                if (continuousFeatures[k])
                {
                    meanAndStdDevTable.put(k, meanAndStdDevTable.getOrDefault(k, new double[4])); // make a new double[] if not already exists
                    if (trainingLabels[i] == 1) meanAndStdDevTable.get(k)[0] += trainingData[i][k]; // add feature value to mean like it
                    else if (trainingLabels[i] == 0) meanAndStdDevTable.get(k)[1] += trainingData[i][k]; // add feature value to mean dislike it
                }
            }
        }

        // calculate probabilities
        double priorLikeProbability = (double) totalLikeCount / trainingData.length;
        double priorDislikeProbability = (double) totalDislikeCount / trainingData.length;

        double likeLikelihood = 1;
        double dislikeLikelihood = 1;

        for (int i = 0; i < testFeature.length; i++)
        {
            if (!continuousFeatures[i])
            {
                // amount of likes so far based on current feature value
                int countingLikes = likeCount.get(i).getOrDefault(testFeature[i], 0);
                // amount of dislikes so far based on current feature value
                int countingDislikes = dislikeCount.get(i).getOrDefault(testFeature[i], 0);

                likeLikelihood *= (double) countingLikes / trainingData.length;
                dislikeLikelihood *= (double) countingDislikes / trainingData.length;
            }
            else
            {
                // calculate means
                double[] table = meanAndStdDevTable.get(i);
                table[0] /= totalLikeCount; // calculate mean for like it
                table[1] /= totalDislikeCount; // calculate mean for dislike it

                double sigmaLikeIt = 0;
                double sigmaDislikeIt = 0;

                // calculate standard deviations
                for (int k = 0; k < trainingData.length; k++)
                {
                    if (trainingLabels[k] == 1) // likes it
                    {
                        sigmaLikeIt += Math.pow(trainingData[k][i] - table[0], 2);
                    }
                    else if (trainingLabels[k] == 0) // dislikes it
                    {
                        sigmaDislikeIt += Math.pow(trainingData[k][i] - table[1], 2);
                    }
                }

                table[2] = Math.sqrt(sigmaLikeIt / (totalLikeCount - 1)); // standard deviation for like it
                table[3] = Math.sqrt(sigmaDislikeIt / (totalDislikeCount - 1)); // standard deviation for dislike it

                likeLikelihood *= Math.exp(-Math.pow((testFeature[i] - table[0]), 2) / (2 * Math.pow(table[2], 2))) / (table[2] * Math.sqrt(2 * Math.PI)); // using gaussian formula
                dislikeLikelihood *= Math.exp(-Math.pow((testFeature[i] - table[1]), 2) / (2 * Math.pow(table[3], 2))) / (table[3] * Math.sqrt(2 * Math.PI)); // using gaussian formula
            }
        }

        // posterior probability = (likelihood * prior probability) / probability of evidence
        // based on p(a|b) = (p(b|a) * p(a)) / (p(b|a) * p(a) + p(b|~a) * p(~a))
        // or p(a|b,c..) = (p(b|a) * p(c|a) * p(a)) / (p(b|a) * p(c|a) * p(a) + p(b|~a) * p(c|~a) * p(~a))
        double evidence = (likeLikelihood * priorLikeProbability) + (dislikeLikelihood * priorDislikeProbability);
        double likePosteriorProbability = (likeLikelihood * priorLikeProbability) / evidence;
        double dislikePosteriorProbability = (dislikeLikelihood * priorDislikeProbability) / evidence;

        return likePosteriorProbability > dislikePosteriorProbability ? 1 : 0;
    }

    /*
        If the current movie that's being checked has the same feature values as the ones in the training dataset then add 1 to the like counter if it's liked
        otherwise add 1 to the dislike counter if it isn't. Then find the probabilities at the end and compare them.

        One of the weaknesses is that all feature values must be the same otherwise it fails.
     */
    static int simpleProbabilityModel(double[][] trainingData, int[] trainingLabels, double[] testFeature)
    {
        // how many movies student X likes based on the features used
        double likeOccurrences = 0.001; // smoothing
        double dislikeOccurrences = 0.001; // smoothing

        for (int i = 0; i < trainingLabels.length; i++)
        {
            if (Arrays.equals(trainingData[i], testFeature)) // if same features
            {
                // increment occurrences
                if (trainingLabels[i] == 1) likeOccurrences++;
                else if (trainingLabels[i] == 0) dislikeOccurrences++;
            }
        }

        // start predicting probability of liking current movie
        double likeProbability = likeOccurrences / (likeOccurrences + dislikeOccurrences);
        double dislikeProbability = dislikeOccurrences / (likeOccurrences + dislikeOccurrences);

        return likeProbability > dislikeProbability ? 1 : 0;
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
        int knnCorrectPredictions = 0;
        int simpleProbCorrectPredictions = 0;
        int bayesCorrectPredictions = 0;
        int gaussianBayesCorrectPredictions = 0;

        for (int i = 0; i < testingData.length; i++)
        {
            boolean knnClassify = knnClassify(trainingData, trainingLabels, testingData[i], 1) == testingLabels[i];
            boolean simpleProbabilities = simpleProbabilityModel(trainingData, trainingLabels, testingData[i]) == testingLabels[i];
            boolean naiveBayesClassify = gaussianNaiveBayesClassify(trainingData, trainingLabels, testingData[i], new boolean[] { false, false }) == testingLabels[i]; // we set all of continuousFeatures array to false so we can use the classical naive bayes calculations
            boolean gaussianNaiveBayesClassify = gaussianNaiveBayesClassify(trainingData, trainingLabels, testingData[i], new boolean[] { true, false }) == testingLabels[i]; // e.g. imdb is continuous, genre is not, hence it'll be { true, false }

            if (knnClassify) knnCorrectPredictions++;
            if (simpleProbabilities) simpleProbCorrectPredictions++;
            if (naiveBayesClassify) bayesCorrectPredictions++;
            if (gaussianNaiveBayesClassify) gaussianBayesCorrectPredictions++;
        }

        double knnAccuracy = (double) knnCorrectPredictions / testingData.length * 100;
        double simpleProbAccuracy = (double) simpleProbCorrectPredictions / testingData.length * 100;
        double bayesAccuracy = (double) bayesCorrectPredictions / testingData.length * 100;
        double gaussianBayesAccuracy = (double) gaussianBayesCorrectPredictions / testingData.length * 100;
        System.out.printf("KNN Accuracy: %.2f%%\n", knnAccuracy);
        System.out.printf("Simple Model Accuracy: %.2f%%\n", simpleProbAccuracy);
        System.out.printf("Naive Bayes Accuracy: %.2f%%\n", bayesAccuracy);
        System.out.printf("Gaussian Naive Bayes Accuracy: %.2f%%\n", gaussianBayesAccuracy);
    }
}
