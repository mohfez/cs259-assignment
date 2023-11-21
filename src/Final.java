//in progress creating template for the project (movie dataset reading, masking, KNN for K=1)
import java.io.*;

public class Final {

	// Use we use 'static' for all methods to keep things simple, so we can call those methods main

	static void Assert (boolean res) // We use this to test our results - don't delete or modify!
	{
	 if(!res)	{
		 System.out.print("Something went wrong.");
	 	 System.exit(0);
	 }
   System.out.println(res);
	}

	// Copy your vector operations here:
	// ...

  static double dot(double[] u, double[] v) {
    double dot = 0;

    for(int i = 0; i < v.length; i++ )  {
      dot += u[i] *  v[i];
    }

    return dot;
  }
 

 static int NumberOfFeatures = 16;
 static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {
 	
 	
     double[] feature = new double[NumberOfFeatures]; 
//     feature[0] = id;  // We use the movie id as a numeric attribute.
    
     switch (genre) { // We also use represent each movie genre as an integer number:
     
         case "Action":  feature[1] = 1; break;
         case "Drama":   feature[2] = 1; break;
         case "Romance": feature[3] = 1; break;
         case "Sci-Fi": feature[4] = 1; break;
         case "Adventure": feature[5] = 1; break;
         case "Horror": feature[6] = 1; break;
         case "Mystery": feature[7] = 1; break;
         case "Thriller": feature[8] = 1; break;
         default: Assert(false);
                  
     }
//     feature[9] = runtime;
//     feature[10] = year;
//     feature[11] = imdb;
//     feature[12] = rt;
//     feature[13] = budget;
//     feature[14] = boxOffice;
     // That is all. We don't use any other attributes for prediction.
     return feature;
 }

 // We are using the dot product to determine similarity:
 static double similarity(double[] u, double[] v) {
    return dot(u, v);  
 }

 // We have implemented KNN classifier for the K=1 case only. You are welcome to modify it to support any K
 static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature) {

	 int bestMatch = -1;
     double bestSimilarity = - Double.MAX_VALUE;  // We start with the worst similarity that we can get in Java.

     for (int i = 0; i < trainingData.length; i++) {
         double currentSimilarity = similarity(testFeature, trainingData[i]);
         if (currentSimilarity > bestSimilarity) {
            bestSimilarity = currentSimilarity;
            bestMatch = i;
         }
     }
     return trainingLabels[bestMatch];
 }

 
 static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels) throws IOException {
     try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
         String line;
         int idx = 0;
         br.readLine(); // skip header line
         while ((line = br.readLine()) != null) {
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

 public static void main(String[] args) {

     double[][] trainingData = new double[100][];
     int[] trainingLabels = new int[100];
     double[][] testingData = new double[100][]; 
     int[] testingLabels = new int[100]; 
     try {
         // You may need to change the path:        	        	
         loadData("./data/training-set.csv", trainingData, trainingLabels);
         loadData("./data/testing-set.csv", testingData, testingLabels);
     } 
     catch (IOException e) {
         System.out.println("Error reading data files: " + e.getMessage());
         return;
     }

     // Compute accuracy on the testing set
     int correctPredictions = 0;

     // Add some lines here: ...
      for(int i = 0; i < testingData.length; i++) {
        if(knnClassify(trainingData, trainingLabels, testingData[i]) == testingLabels[i]) {
          correctPredictions++;
        }
      }
     
     double accuracy = (double) correctPredictions / testingData.length * 100;
     System.out.printf("A: %.2f%%\n", accuracy);
     
 }

}




