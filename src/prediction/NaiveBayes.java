package prediction;

public class NaiveBayes extends GaussianNaiveBayes
{
    @Override
    public double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice)
    {
        double[] feature = new double[1]; // n features

        switch (genre)
        {
            case "Action":  feature[0] = 0; break;
            case "Drama":   feature[0] = 1; break;
            case "Romance": feature[0] = 2; break;
            case "Sci-Fi": feature[0] = 3; break;
            case "Adventure": feature[0] = 4; break;
            case "Horror": feature[0] = 5; break;
            case "Mystery": feature[0] = 6; break;
            case "Thriller": feature[0] = 7; break;
        }

        return feature;
    }
}
