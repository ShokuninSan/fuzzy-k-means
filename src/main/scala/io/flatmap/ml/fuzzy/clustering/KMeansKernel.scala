package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{Axis, sum, DenseMatrix}
import io.flatmap.ml.fuzzy.functions._

trait KMeansKernel {

  val m: Int
  val c: Int

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 352, eq. 10.30
    */
  def calculateCentroids(data: DenseMatrix[Double], u: DenseMatrix[Double], m: Int): DenseMatrix[Double] = {
    val um = pow(u, m)
    (um * data) / (um * DenseMatrix.ones[Double](data.rows, data.cols))
  }

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 353, eq. 10.32a
    */
  def updateMemberships(c: Int, u: DenseMatrix[Double], d: DenseMatrix[Double], m: Int): DenseMatrix[Double] = {
    val _u = pow(d, -2 / (m - 1))
    val ones = DenseMatrix.ones[Double](c, 1)
    _u / (ones * sum(_u, Axis._0).inner.toDenseMatrix)
  }

  def run(data: DenseMatrix[Double], centroids: Option[DenseMatrix[Double]] = None, errorThreshold: Double =  0.005, maxIterations: Int = 1000): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // step 1: c and m are already fixed. Initialize the partition matrix and r
    var u = initGaussian(c, data.rows) // classes x samples
    var r = 0
    var v = DenseMatrix.zeros[Double](1, c)

    while (r < maxIterations) {
      val u2 = u.copy

      // step 2: calculate cluster centers (eq 10.30)
      v = centroids getOrElse calculateCentroids(data, u, m)

      // step 3: update partition matrix
      val d = distance(v, data)
      u = updateMemberships(c, u, d, m)

      // step 4: calculate the Frobenius norm of the two successive fuzzy partitions
      if (norm(u - u2) <= errorThreshold)
        r = maxIterations
      else
        r += 1
    }

    (v, u)
  }

}
