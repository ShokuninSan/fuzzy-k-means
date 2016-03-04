package io.flatmap.ml.fuzzy

import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.linalg.DenseMatrix
import breeze.linalg.functions.euclideanDistance

package object functions {

  def distance(data: DenseMatrix[Double], centroids: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(data.cols == centroids.cols, message = "invalid dimensions for matrices (data.cols should equal to centroids.cols)")
    val d = DenseMatrix.zeros[Double](data.rows, centroids.rows)
    for {
      point <- 0 to data.rows - 1
      centroid <- 0 to centroids.rows - 1
    } yield d(point, centroid) = euclideanDistance(data(point, ::).inner, centroids(centroid, ::).inner)
    d
  }

  def initGaussian(n_samples: Int, n_features: Int): DenseMatrix[Double] =
    DenseMatrix.rand(n_samples, n_features, breeze.stats.distributions.Gaussian(0, 1))

  def norm(x: DenseMatrix[Double]): Double = breeze.linalg.norm(x.toDenseVector)

  def pow(x: DenseMatrix[Double], exp: Double): DenseMatrix[Double] =
    breeze.numerics.pow(x.toDenseVector, exp).toDenseMatrix.reshape(x.rows, x.cols)

  def closeTo(a: DenseVector[Double], b: DenseVector[Double], relDiff: Double = 1e-2): Boolean = {
    assert(a.length == b.length)
    a.toArray zip b.toArray forall { ab => breeze.numerics.closeTo(ab._1, ab._2, relDiff) }
  }

}
