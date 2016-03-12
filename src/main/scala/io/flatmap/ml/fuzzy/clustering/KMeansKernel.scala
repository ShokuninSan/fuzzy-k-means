package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{Axis, sum, DenseMatrix}
import io.flatmap.ml.fuzzy.functions._

trait KMeansKernel {

  val fuzziness: Double
  val numClusters: Int

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 352, eq. 10.30
    */
  def calculateCentroids(data: DenseMatrix[Double], u: DenseMatrix[Double]): DenseMatrix[Double] = {
    val um = pow(u, fuzziness)
    (um * data) / (um * DenseMatrix.ones[Double](data.rows, data.cols))
  }

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 353, eq. 10.32a
    */
  def updateMemberships(u: DenseMatrix[Double], d: DenseMatrix[Double]): DenseMatrix[Double] = {
    val _u = pow(d, -2 / (fuzziness - 1))
    val ones = DenseMatrix.ones[Double](numClusters, 1)
    fmax(_u /= (ones * sum(_u, Axis._0).inner.toDenseMatrix), eps)
  }

  def initPartitionMatrix(rows: Int, cols: Int): DenseMatrix[Double] = {
    val u0 = DenseMatrix.rand[Double](rows, cols) // classes x samples
    val u: DenseMatrix[Double] = u0 / (DenseMatrix.ones[Double](rows, 1) * sum(u0, Axis._0).inner.toDenseMatrix)
    fmax(u, eps)
  }

  def initClusterCentroids(numClusters: Int): DenseMatrix[Double] = DenseMatrix.zeros[Double](1, numClusters)

  def run(data: DenseMatrix[Double], centroids: Option[DenseMatrix[Double]] = None, errorThreshold: Double =  0.005, maxIterations: Int = 1000): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // step 1: initialize the partition matrix and r
    var u = initPartitionMatrix(numClusters, data.rows)
    var v = initClusterCentroids(numClusters)
    var r = 0

    while (r < maxIterations) {
      val u2 = u.copy

      // step 2: calculate cluster centers (eq 10.30); prediction uses centroids given as parameter
      v = centroids getOrElse calculateCentroids(data, u)

      // step 3: update partition matrix
      u = updateMemberships(u, distance(v, data))

      // step 4: calculate the Frobenius norm of the two successive fuzzy partitions
      if (norm(u - u2) <= errorThreshold)
        r = maxIterations
      else
        r += 1
    }

    (v, u)
  }

}
