package prediction;

import java.util.Arrays;

public class SimpleProbabilities extends PredictionModel
{
    @Override
    public double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice)
    {
        double[] feature = new double[2]; // n features

        feature[0] = imdb;
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

        return feature;
    }

    /*
        If the current movie that's being checked has the same feature values as the ones in the training dataset then add 1 to the like counter if it's liked
        otherwise add 1 to the dislike counter if it isn't. Then find the probabilities at the end and compare them.

        One of the weaknesses is that all feature values must be the same otherwise it fails.
     */
    public int simpleProbabilityModel(int testingDataIndex)
    {
        // how many movies student X likes based on the features used
        double likeOccurrences = 0.001; // smoothing
        double dislikeOccurrences = 0.001; // smoothing

        for (int i = 0; i < trainingLabels.length; i++)
        {
            if (Arrays.equals(trainingData[i], testingData[testingDataIndex])) // if same features
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
}
