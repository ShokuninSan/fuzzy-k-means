package io.flatmap.ml.fuzzy

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.functions.euclideanDistance
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix

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
    * Calculate the Euclidean distance between two Vectors.
    *
    * See also: [[https://en.wikipedia.org/wiki/Euclidean_distance#definition]].clone(): AnyRef = super.clone()
    *
    * @param a
    * @param b
    * @return distance
    */
  def distance(a: Vector, b: Vector): Double =
    math.sqrt(a.toArray.zip(b.toArray).map(v => v._1 - v._2).map(x => x * x).sum)

  /**
    * Calculate the distance between data points and cluster centroids
    *
    * @param data
    * @param centroids
    * @return Matrix of shape (#datapoints x #centroids)
    */
  def distance(data: RowMatrix, centroids: RowMatrix): RowMatrix = {
    assert(data.numCols() == centroids.numCols(), message = "invalid dimensions for matrices (data.cols should equal to centroids.cols)")
    val c = centroids.rows.collect()
    new RowMatrix(
      data.rows.map(d => Vectors.dense(c.map(c => distance(d, c)))),
      data.numRows(),
      centroids.numRows().toInt)
  }

  /**
    * Calculate the Frobenius norm of a DenseMatrix
    *
    * @param x Matrix
    * @return Double
    */
  def norm(x: DenseMatrix[Double]): Double = breeze.linalg.norm(x.toDenseVector)

  /**
    * Calculate the Frobenius norm of a RowMatrix
    *
    * See also [[https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm]].
    *
    * @param x Matrix
    * @return Double
    */
  def norm(x: RowMatrix): Double =
    math.sqrt(x.rows.map(v => v.toArray.map(x => math.pow(math.abs(x), 2)).reduce(_+_)).reduce(_+_))

  /**
    * Calculate the power of a DenseMatrix
    *
    * @param x Matrix
    * @param exp The exponent
    * @return Matrix
    */
  def pow(x: DenseMatrix[Double], exp: Double): DenseMatrix[Double] = breeze.numerics.pow(x.toDenseVector, exp).toDenseMatrix.reshape(x.rows, x.cols)

  /**
    * Calculate the power of a RowMatrix
    *
    * @param matrix Matrix
    * @param exp The exponent
    * @return Matrix
    */
  def pow(matrix: RowMatrix, exp: Double): RowMatrix =
    new RowMatrix(
      matrix.rows.map(v => Vectors.dense(v.toArray.map(x => math.pow(math.abs(x), exp)))),
      matrix.numRows,
      matrix.numCols.toInt)

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
  def allOnesMatrix(rows: Int, cols: Int): DenseMatrix[Double] = DenseMatrix.ones[Double](rows, cols)

}
