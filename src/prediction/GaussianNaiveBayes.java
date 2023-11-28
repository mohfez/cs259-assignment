package prediction;

import java.util.HashMap;
import java.util.Map;

public class GaussianNaiveBayes extends PredictionModel
{
    @Override
    public double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice)
    {
        double[] feature = new double[5]; // n features

        feature[0] = year;
        switch (genre)
        {
            case "Action":  feature[1] = 0; break;
            case "Drama":   feature[1] = 1; break;
            case "Romance": feature[1] = 2; break;
            case "Sci-Fi": feature[1] = 3; break;
            case "Adventure": feature[1] = 4; break;
            case "Horror": feature[1] = 5; break;
            case "Mystery": feature[1] = 6; break;
            case "Thriller": feature[1] = 7; break;
        }
        feature[2] = budget;
        feature[3] = runtime;
        feature[4] = boxOffice;

        return feature;
    }

    /*
        Since the dataset has many continuous variables, gaussian naive bayes might be more appropriate instead of the classical.
        runtime, imdb, rt, budget, box office all look like good features that are continuous.
        genre and year look like they're categorical.

        Note: set all continuousFeatures array to false if you want to use the classical naive bayes model

        The first part of the code makes tables using hashsets, for example let's use genre and year as our features:

        ["likeCount" hashmap]
        +-------------------+---------------+---------+
        | Feature 0(Genre)  | Feature Value | Like It |
        +-------------------+---------------+---------+
        |                 - | Action        |      12 |
        |                 - | Drama         |       5 |
        |                 - | Romance       |       9 |
        |                 - | Sci-Fi        |      12 |
        |                 - | Adventure     |      15 |
        |                 - | Horror        |       4 |
        |                 - | Mystery       |       0 |
        |                 - | Thriller      |       4 |
        +-------------------+---------------+---------+

        +------------------+---------------+---------+
        | Feature 1(Year)  | Feature Value | Like It |
        +------------------+---------------+---------+
        |                - | 2021          |      18 |
        |                - | 2022          |      18 |
        |                - | 2023          |      25 |
        +------------------+---------------+---------+

        ["dislikeCount" hashmap]
        +-------------------+---------------+------------+
        | Feature 0(Genre)  | Feature Value | Dislike It |
        +-------------------+---------------+------------+
        |                 - | Action        |          3 |
        |                 - | Drama         |         10 |
        |                 - | Romance       |          6 |
        |                 - | Sci-Fi        |          3 |
        |                 - | Adventure     |          5 |
        |                 - | Horror        |          6 |
        |                 - | Mystery       |          5 |
        |                 - | Thriller      |          1 |
        +-------------------+---------------+------------+

        +------------------+---------------+------------+
        | Feature 1(Year)  | Feature Value | Dislike It |
        +------------------+---------------+------------+
        |                - | 2021          |         19 |
        |                - | 2022          |         15 |
        |                - | 2023          |          5 |
        +------------------+---------------+------------+

        we then use these values to calculate the probabilities.

        for continuous features we use a hashmap to make a table of all the means and standard deviations, for example; using budget, runtime, boxOffice
        +----------------+--------------------+-------------------+--------------------+----------------------+
        |   Feature i    |     Mean Like      |   Mean Dislike    | Std. Dev For Like  | Std. Dev For Dislike |
        +----------------+--------------------+-------------------+--------------------+----------------------+
        | 2 (Budget)     |                111 | 77.76923076923077 | 43.282791037547476 |    43.80546801873119 |
        | 3 (Runtime)    | 111.72131147540983 | 105.7948717948718 |  9.181741206584917 |    9.811860404041328 |
        | 4 (Box Office) |  170.8360655737705 | 116.3076923076923 | 158.10283365875816 |   128.63026447641028 |
        +----------------+--------------------+-------------------+--------------------+----------------------+

        we use these values for the gaussian naive bayes formula.
     */
    public int gaussianNaiveBayesClassify(int testingDataIndex, boolean[] continuousFeatures)
    {
        // total likes and dislikes
        int totalLikeCount = 0;
        int totalDislikeCount = 0;

        // hashmaps of feature i : (feature val, (dis)like count)
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

        for (int i = 0; i < testingData[testingDataIndex].length; i++)
        {
            if (!continuousFeatures[i])
            {
                // amount of likes so far based on current feature value
                int countingLikes = likeCount.get(i).getOrDefault(testingData[testingDataIndex][i], 0);
                // amount of dislikes so far based on current feature value
                int countingDislikes = dislikeCount.get(i).getOrDefault(testingData[testingDataIndex][i], 0);

                likeLikelihood *= (double) countingLikes / totalLikeCount;
                dislikeLikelihood *= (double) countingDislikes / totalDislikeCount;
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

                likeLikelihood *= Math.exp(-Math.pow((testingData[testingDataIndex][i] - table[0]), 2) / (2 * Math.pow(table[2], 2))) / (table[2] * Math.sqrt(2 * Math.PI)); // using gaussian formula
                dislikeLikelihood *= Math.exp(-Math.pow((testingData[testingDataIndex][i] - table[1]), 2) / (2 * Math.pow(table[3], 2))) / (table[3] * Math.sqrt(2 * Math.PI)); // using gaussian formula
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
}
