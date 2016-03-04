package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{Axis, sum, DenseMatrix}
import io.flatmap.ml.fuzzy.functions.pow

trait KMeansKernel {

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 352, eq. 10.30
    */
  def calculateCentroids(data: DenseMatrix[Double], u: DenseMatrix[Double], m: Int): DenseMatrix[Double] = {
    val um = pow(u, m)
    val num = (um * data)
    val denom = (um * DenseMatrix.ones[Double](data.rows, data.cols))
    num / denom
  }

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 353, eq. 10.32a
    */
  def updateMemberships(c: Int, u: DenseMatrix[Double], d: DenseMatrix[Double], m: Int): DenseMatrix[Double] = {
    val _u = pow(d, -2 / (m - 1))
    val ones = DenseMatrix.ones[Double](c, 1)
    (_u / (ones * sum(_u, Axis._0).inner.toDenseMatrix))
  }

}
