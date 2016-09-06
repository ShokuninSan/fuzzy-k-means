package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand
import io.flatmap.ml.fuzzy.clustering.kernels.KMeansKernel
import io.flatmap.ml.fuzzy.numerics._
import io.flatmap.ml.normalization.{DenseMatrixMeanNormalizer, Normalizer}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

trait SparkIterativeOptimization {

  val numClusters: Int

  def initMembershipRowMatrix(numSamples: Int, numClusters: Int)(implicit sc: SparkContext): RowMatrix =
    new RowMatrix(
      sc.parallelize(Seq.fill(numSamples)(numClusters)).map(elms => Vectors.dense(DenseVector.rand(elms, Rand.uniform).toArray)),
      numSamples,
      numClusters
    )

  def run(data: RowMatrix, clusterCentroids: Option[org.apache.spark.mllib.linalg.DenseMatrix] = None, errorThreshold: Double =  0.005, maxIterations: Int = 2000, fuzziness: Double = 2.0)(implicit kernel: KMeansKernel[RowMatrix, org.apache.spark.mllib.linalg.DenseMatrix, Normalizer[RowMatrix]], sc: SparkContext): (org.apache.spark.mllib.linalg.DenseMatrix, RowMatrix) = {
    // step 1: initialization
    var memberships = initMembershipRowMatrix(numClusters, data.numRows().toInt)
    var centroids = org.apache.spark.mllib.linalg.DenseMatrix.zeros(numClusters, data.numCols().toInt)
    var iteration = 0

    while (iteration < maxIterations) {
      val previousMemberships = new RowMatrix(memberships.rows, memberships.numRows(), memberships.numCols().toInt)

      // step 2: calculate cluster centers (eq 10.30); prediction uses centroids given as parameter
      centroids = clusterCentroids getOrElse kernel.calculateCentroids(data, memberships, fuzziness)

      // step 3: calculate memberhsip matrix (equation 10.32a)
      memberships = kernel.calculateMemberships(distance(data, centroids), fuzziness)

      // step 4: calculate the norm of the two successive fuzzy partitions
      if (norm(sub(memberships, previousMemberships)) <= errorThreshold)
        iteration = maxIterations
      else
        iteration += 1
    }

    (centroids, memberships)
  }

}

trait BreezeIterativeOptimization {

  val numClusters: Int

  /**
    * Initialize membership matrix
    *
    * This method randomly initializes a numSamples x numClusters matrix with uniformly distributed values which are normalized.
    *
    * @param numSamples # of datapoints (samples)
    * @param numClusters # of centroids or clusters
    * @return Matrix of membership degrees of data points to clusters of shape (#datapoints x #clusters)
    */
  def initMembershipMatrix(numSamples: Int, numClusters: Int): DenseMatrix[Double] = {
    val _u = DenseMatrix.rand[Double](numSamples, numClusters, rand = Rand.uniform)
    DenseMatrixMeanNormalizer.normalize(_u)
  }

  /**
    * Run the Fuzzy-k-Means algorithm
    *
    * This method runs the Fuzzy-k-Means algorithm according to Timothy Ross book "Fuzzy Logic with Engineering Applications", p. 352-353,
    * which is a so-called iterative optimization algorithm (Ross, 2010, p.352).
    *
    * Ross only outlined the training of a Fuzzy-k-Means estimator. However, this implementation is generic in the sense that when
    * clusterCentroids are passed to this method, the same implementation can also be used for prediction of new (unseen) data points.
    *
    * @param data Input data to be clustered
    * @param clusterCentroids Used for prediction where clusters are fixed
    * @param errorThreshold The prescribed level of accuracy
    * @param maxIterations Maximum number of iteration to run
    * @return Tuple with calculated centroids and memberships
    */
  def run(data: DenseMatrix[Double], clusterCentroids: Option[DenseMatrix[Double]] = None, errorThreshold: Double =  0.005, maxIterations: Int = 2000, fuzziness: Double = 2.0)(implicit kernel: KMeansKernel[DenseMatrix[Double], DenseMatrix[Double], Normalizer[DenseMatrix[Double]]]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // step 1: initialization
    var memberships = initMembershipMatrix(numClusters, data.rows)
    var centroids = DenseMatrix.zeros[Double](numClusters, data.cols)
    var iteration = 0

    while (iteration < maxIterations) {
      val previousMemberships = memberships.copy

      // step 2: calculate cluster centers (eq 10.30); prediction uses centroids given as parameter
      centroids = clusterCentroids getOrElse kernel.calculateCentroids(data, memberships, fuzziness)

      // step 3: calculate memberhsip matrix (equation 10.32a)
      memberships = kernel.calculateMemberships(distance(centroids, data), fuzziness)

      // step 4: calculate the norm of the two successive fuzzy partitions
      if (norm(memberships - previousMemberships) <= errorThreshold)
        iteration = maxIterations
      else
        iteration += 1
    }

    (centroids, memberships)
  }

}
