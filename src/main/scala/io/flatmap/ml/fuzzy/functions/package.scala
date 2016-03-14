package io.flatmap.ml.fuzzy

import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.linalg.DenseMatrix
import breeze.linalg.functions.euclideanDistance
import breeze.stats.distributions.RandBasis
import com.github.fommil.netlib.LAPACK.{getInstance=>lapack}

package object functions {

  /**
    * The machine epsilon
    *
    * This implementation is borrowed from the great Breeze library. See [[breeze.linalg.rank]] (line 29).
    */
  lazy val eps: Double = 2.0 * lapack.dlamch("e")

  /**
    * Calculate the distance between data points and cluster centroids
    *
    * @param data
    * @param centroids
    * @return Matrix of shape (#datapoints x #centroids)
    */
  def distance(data: DenseMatrix[Double], centroids: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(data.cols == centroids.cols, message = "invalid dimensions for matrices (data.cols should equal to centroids.cols)")
    val d = DenseMatrix.zeros[Double](data.rows, centroids.rows)
    for {
      point <- 0 to data.rows - 1
      centroid <- 0 to centroids.rows - 1
    } yield d(point, centroid) = euclideanDistance(data(point, ::).inner, centroids(centroid, ::).inner)
    d
  }

  /**
    * Calculate the Frobenius norm of a matrix
    *
    * @param x Matrix
    * @return Double
    */
  def norm(x: DenseMatrix[Double]): Double = breeze.linalg.norm(x.toDenseVector)

  /**
    * Calculate the power of a matrix
    *
    * @param x Matrix
    * @param exp The exponent
    * @return Matrix
    */
  def pow(x: DenseMatrix[Double], exp: Double): DenseMatrix[Double] = breeze.numerics.pow(x.toDenseVector, exp).toDenseMatrix.reshape(x.rows, x.cols)

  /**
    * Calculate if two given matrices are close
    *
    * This function wraps the [[breeze.numerics.closeTo]] function to conveniently use it for matrices as well.
    *
    * @param a Matrix
    * @param b Matrix
    * @param epsilon The accuracy level of closeness
    * @return True if matrices are close enough, False otherwise
    */
  def closeTo(a: DenseVector[Double], b: DenseVector[Double], epsilon: Double = 1e-2): Boolean = {
    assert(a.length == b.length)
    a.toArray zip b.toArray forall { ab => breeze.numerics.closeTo(ab._1, ab._2, epsilon) }
  }

  /**
    * Element-wise maximum of array elements
    *
    * This function is imitated from the Numpy library. Compare two matrices and return a new matrix containing the element-wise maxima. If
    * one of the elements being compared is a NaN, the value of parameter v is used.
    *
    * @param x Matrix
    * @param v Value for NaN replacement
    * @return Matrix wihout NaN values
    */
  def fmax(x: DenseMatrix[Double], v: Double): DenseMatrix[Double] = x.mapValues(x => if (x.isNaN) v else x)

  /**
    * Create a matrix of all ones
    *
    * @param rows
    * @param cols
    * @return Matrix of all ones
    */
  def unitMatrix(rows: Int, cols: Int): DenseMatrix[Double] = DenseMatrix.ones[Double](rows, cols)

}
