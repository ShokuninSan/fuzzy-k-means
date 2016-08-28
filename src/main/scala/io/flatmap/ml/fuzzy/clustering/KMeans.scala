package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.clustering.kernels.KMeansKernel
import io.flatmap.ml.fuzzy.numerics._
import io.flatmap.ml.normalization.Normalizer

private[fuzzy] case class KMeansModel(val numClusters: Int, val fuzziness: Double, val epsilon: Double, centroids: DenseMatrix[Double], u: DenseMatrix[Double]) extends IterativeOptimization {

  def predict(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000)(implicit kernel: KMeansKernel[DenseMatrix[Double], DenseMatrix[Double], Normalizer[DenseMatrix[Double]]]): DenseMatrix[Double] =
    run(data, Some(centroids), errorThreshold, maxIterations) match {
      case (_, memberships) => memberships
    }

}

case class KMeans(val numClusters: Int, val fuzziness: Double, val epsilon: Double = eps) extends IterativeOptimization {

  def fit(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000)(implicit kernel: KMeansKernel[DenseMatrix[Double], DenseMatrix[Double], Normalizer[DenseMatrix[Double]]]): KMeansModel =
    run(data, errorThreshold = errorThreshold, maxIterations = maxIterations) match {
      case (centroids, memberships) => KMeansModel(numClusters, fuzziness, epsilon, centroids, memberships)
    }

}
