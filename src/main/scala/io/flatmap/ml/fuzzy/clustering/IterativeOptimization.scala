package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Rand
import io.flatmap.ml.fuzzy.clustering.kernels.KMeansKernel
import io.flatmap.ml.fuzzy.numerics._
import io.flatmap.ml.normalization.MeanNormalizer

trait IterativeOptimization {

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
    MeanNormalizer.normalize(_u)
  }

  /**
    * Initialize cluster centroids
    *
    * @param numClusters
    * @return Matrix of shape 1 x numClusters
    */
  def initClusterCentroids(numClusters: Int, numFeatures: Int): DenseMatrix[Double] = DenseMatrix.zeros[Double](numClusters, numFeatures)

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
  def run(data: DenseMatrix[Double], clusterCentroids: Option[DenseMatrix[Double]] = None, errorThreshold: Double =  0.005, maxIterations: Int = 2000, fuzziness: Double = 2.0)(implicit kernel: KMeansKernel): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // step 1: initialization
    var memberships = initMembershipMatrix(numClusters, data.rows)
    var centroids = initClusterCentroids(numClusters, data.cols)
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
