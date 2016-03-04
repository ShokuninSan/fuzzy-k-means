package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix

private[fuzzy] case class KMeansModel(val numClusters: Int, val fuzziness: Int, centroids: DenseMatrix[Double], u: DenseMatrix[Double]) extends KMeansKernel {

  def predict(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000): DenseMatrix[Double] =
    run(data, Some(centroids), errorThreshold, maxIterations) match {
      case (_, memberships) => memberships
    }

}

case class KMeans(val numClusters: Int, val fuzziness: Int) extends KMeansKernel {

  def fit(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000): KMeansModel =
    run(data, errorThreshold = errorThreshold, maxIterations = maxIterations) match {
      case (centroids, memberships) => KMeansModel(numClusters, fuzziness, centroids, memberships)
    }

}
