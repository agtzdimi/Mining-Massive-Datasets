package UnsupervisedModels

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * @author Dimitris Agtzidis - Argyrios Vartholomaios
  * @param sc Spark Context instance
  * @param dim1 Values for first Dimension
  * @param dim2 Values for Second Dimension
  * @param k number of clusters we want at the end
  * @param maxIter Maximum iterations to be run
  */
class KMeans(sc: SparkContext, dim1: RDD[Double], dim2: RDD[Double], k:Int, maxIter:Int) {

  def run(outputPath: String = "output/kmeans") = {
    val samples = dim1.takeSample(withReplacement = false, k).zip(dim2.takeSample(withReplacement = false, k))
    var centroids: RDD[(Int, (Double, Double))] = sc.parallelize(samples.zipWithIndex).map(_.swap)
    val values: RDD[(Double, Double)] = dim1.zip(dim2)
    var i = 0
    var clusters: RDD[(Int, (Double, Double))] = null
    while (i < maxIter) {
      i += 1
      clusters = getClusters(centroids, values)
      centroids = getCentroids(clusters, centroids)
    }
    clusters.map(x => x._1 + "," +  x._2._1 + "," + x._2._2).saveAsTextFile(outputPath)
  }

  /**
    * Given the centroids and data points, it assigns clusters to each data point
    * by calculating the distance to each centroid and choosing the cluster
    * represented by the closest centroid
    * @param centroids Centroids for quiet, medium and loud clusters
    * @param songData Song data records
    * @return each data point with cluster assigned
    */
  def getClusters(centroids: RDD[(Int, (Double, Double))], songData: RDD[(Double, Double)]): RDD[(Int, (Double, Double))] = {
    val c : Array[(Int, (Double, Double))] = centroids.collect
    songData.map(x => (c.map(c => (c._1, math.sqrt(math.pow(c._2._1 - x._1, 2) + math.pow(c._2._2 - x._2, 2))))
      .minBy(y => y._2)._1, x))
  }

  /**
    * Given assigned clusters and centroids, calculates new centroids by getting the mean for
    * each clusters
    * @param clusters
    * @param centroids
    * @return
    */
  def getCentroids(clusters: RDD[(Int, (Double, Double))], centroids: RDD[(Int, (Double, Double))]): RDD[(Int, (Double, Double))] = {
    clusters
      // We do a full outer join to include centroids that don't have any values in it
      .fullOuterJoin(centroids)
      // Combine by cluster ID and calculate means for the clusters
      .combineByKey[((Double, Double), (Int, Int))](
      // Create combiner
      // checks whether there is a Song Record available,
      //  if so it emits the loudness with count 1
      // Otherwise it emits 0 with count 1
      (v: (Option[(Double, Double)], Option[(Double, Double)])) =>
        ((if (v._1.isEmpty) 0 else v._1.get._1, if (v._1.isEmpty) 0 else v._1.get._2), (1,1)),

      // Merge Value
      // This function similarly merges loudness values that we've seen before in the same partition
      (acc: ((Double, Double), (Int, Int)), v : (Option[(Double, Double)], Option[(Double, Double)]))
      => ((acc._1._1 + (if (v._1.isEmpty) 0 else v._1.get._1), acc._1._2 + (if (v._1.isEmpty) 0 else v._1.get._2)),
        (acc._2._1 + 1, acc._2._2 + 1)),

      // Merge Combiners
      // Merges values for loudness across partitions
      (acc1: ((Double, Double), (Int, Int)), acc2: ((Double, Double), (Int, Int))) =>
        ((acc1._1._1 + acc2._1._1, acc1._1._2 + acc2._1._2), (acc1._2._1 + acc2._2._1, acc1._2._2 + acc2._2._2))
    ) // Calculate Mean for the clusters
      .mapValues(x => (x._1._1/ x._2._1.toFloat, x._1._2 / x._2._2.toFloat))
  }

}
