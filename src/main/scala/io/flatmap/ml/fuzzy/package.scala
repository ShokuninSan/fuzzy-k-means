package io.flatmap.ml

import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.linalg.DenseMatrix
import breeze.linalg.functions.euclideanDistance

package object fuzzy {

  def distance(data: DenseMatrix[Double], centroids: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(data.cols == centroids.cols, message = "invalid dimensions for matrices (data.cols should equal to centroids.cols)")
    val d = DenseMatrix.zeros[Double](data.rows, centroids.rows)
    for {
      point <- 0 to data.rows - 1
      centroid <- 0 to centroids.rows - 1
    } yield {
      val x = data(point, ::).inner
      val v = centroids(centroid, ::).inner
      d(point, centroid) = euclideanDistance(x, v)
    }
    d
  }

}
