package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.functions.pow

trait KMeansKernel {

  /**
    * Ross T. (2010), Fuzzy Logic with Engineering Applications, p. 352, eq. 10.30
    */
  def calculateCentroids(data: DenseMatrix[Double], um: DenseMatrix[Double], m: Int): DenseMatrix[Double] =
    (um * data) / (um * DenseMatrix.ones[Double](data.rows, data.cols))

}
