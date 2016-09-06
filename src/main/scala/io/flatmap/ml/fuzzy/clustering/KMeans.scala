package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.clustering.kernels.KMeansKernel
import io.flatmap.ml.fuzzy.numerics._
import io.flatmap.ml.normalization.Normalizer
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix

private[fuzzy] case class KMeansModel(val numClusters: Int, val fuzziness: Double, val epsilon: Double, centroids: DenseMatrix[Double], u: DenseMatrix[Double]) extends BreezeIterativeOptimization {

  def predict(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000)(implicit kernel: KMeansKernel[DenseMatrix[Double], DenseMatrix[Double], Normalizer[DenseMatrix[Double]]]): DenseMatrix[Double] =
    run(data, Some(centroids), errorThreshold, maxIterations) match {
      case (_, memberships) => memberships
    }

}

case class KMeans(val numClusters: Int, val fuzziness: Double, val epsilon: Double = eps) extends BreezeIterativeOptimization {

  def fit(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000)(implicit kernel: KMeansKernel[DenseMatrix[Double], DenseMatrix[Double], Normalizer[DenseMatrix[Double]]]): KMeansModel =
    run(data, errorThreshold = errorThreshold, maxIterations = maxIterations) match {
      case (centroids, memberships) => KMeansModel(numClusters, fuzziness, epsilon, centroids, memberships)
    }

}

private[fuzzy] case class SparkKMeansModel(val numClusters: Int, val fuzziness: Double, val epsilon: Double, centroids: org.apache.spark.mllib.linalg.DenseMatrix, u: RowMatrix) extends SparkIterativeOptimization {

  def predict(data: RowMatrix, errorThreshold: Double =  0.005, maxIterations: Int = 1000)(implicit kernel: KMeansKernel[RowMatrix, org.apache.spark.mllib.linalg.DenseMatrix, Normalizer[RowMatrix]], sc: SparkContext): RowMatrix =
    run(data, Some(centroids), errorThreshold, maxIterations) match {
      case (_, memberships) => memberships
    }

}

case class SparkKMeans(val numClusters: Int, val fuzziness: Double, val epsilon: Double = eps) extends SparkIterativeOptimization {

  def fit(data: RowMatrix, errorThreshold: Double =  0.005, maxIterations: Int = 1000)(implicit kernel: KMeansKernel[RowMatrix, org.apache.spark.mllib.linalg.DenseMatrix, Normalizer[RowMatrix]], sc: SparkContext): SparkKMeansModel =
    run(data, errorThreshold = errorThreshold, maxIterations = maxIterations) match {
      case (centroids, memberships) => SparkKMeansModel(numClusters, fuzziness, epsilon, centroids, memberships)
    }

}