package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{Axis, sum, DenseMatrix}
import breeze.stats.distributions.Rand
import io.flatmap.ml.fuzzy.functions._

trait KMeansKernel {

  val fuzziness: Double
  val numClusters: Int
  val epsilon: Double

  /**
    * Calculate cluster centroids
    *
    * This is a vectorized implementation according to Timothy Ross book "Fuzzy Logic with Engineering Applications", p. 352,
    * equation (10.30)
    *
    * @param data Matrix of samples of shape (#datapoints x #features)
    * @param memberships Matrix of membership degrees of data points to clusters of shape (#centroids x #datapoints)
    * @return Matrix of shape #clusters x #features
    */
  def calculateCentroids(data: DenseMatrix[Double], memberships: DenseMatrix[Double]): DenseMatrix[Double] = {
    val _u = pow(memberships, fuzziness)
    val ones = unitMatrix(data.rows, data.cols)
    (_u * data) / (_u * ones)
  }

  /**
    * Calculate cluster membership of data points
    *
    * This is a vectorized implementation according to Timothy Ross book "Fuzzy Logic with Engineering Applications", p. 353,
    * equation (10.32a)
    *
    * @param distances Matrix of distances between centroids and data points of shape (#datapoints x #centroids)
    * @return Matrix of calculated membership degrees of shape (#clusters x #datapoints)
    */
  def calculateMemberships(distances: DenseMatrix[Double]): DenseMatrix[Double] = {
    val _u = pow(distances, -2 / (fuzziness - 1))
    normalize(_u)
  }

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
    normalize(_u)
  }

  /**
    * Normalizes the values of a matrix
    *
    * This method normalizes each value over the sum of the respective column. Before the matrix is returned, NaN (Not a Number) values are
    * substituted with epsilon.
    *
    * @param m Matrix (e.g. memberships or distances)
    * @return Normalized matrix
    */
  def normalize(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ones: DenseMatrix[Double] = unitMatrix(m.rows, 1)
    val sumByColumns: DenseMatrix[Double] = sum(m, Axis._0).inner.toDenseMatrix // shape (1 x #columns)
    val normalized: DenseMatrix[Double] = m / (ones * sumByColumns) // '/' calculates element-wise, i.e. (#rows x #cols) / (#rows x #cols)
    fmax(normalized, epsilon) // substitute NaNs by epsilon
  }

  /**
    * Initialize cluster centroids
    *
    * @param numClusters
    * @return Matrix of shape 1 x numClusters
    */
  def initClusterCentroids(numClusters: Int): DenseMatrix[Double] = DenseMatrix.zeros[Double](1, numClusters)

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
  def run(data: DenseMatrix[Double], clusterCentroids: Option[DenseMatrix[Double]] = None, errorThreshold: Double =  0.005, maxIterations: Int = 2000): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // step 1: initialization
    var memberships = initMembershipMatrix(numClusters, data.rows)
    var centroids = initClusterCentroids(numClusters)
    var iteration = 0

    while (iteration < maxIterations) {
      val previousMemberships = memberships.copy

      // step 2: calculate cluster centers (eq 10.30); prediction uses centroids given as parameter
      centroids = clusterCentroids getOrElse calculateCentroids(data, memberships)

      // step 3: calculate memberhsip matrix (equation 10.32a)
      memberships = calculateMemberships(distance(centroids, data))

      // step 4: calculate the norm of the two successive fuzzy partitions
      if (norm(memberships - previousMemberships) <= errorThreshold)
        iteration = maxIterations
      else
        iteration += 1
    }

    (centroids, memberships)
  }

}
