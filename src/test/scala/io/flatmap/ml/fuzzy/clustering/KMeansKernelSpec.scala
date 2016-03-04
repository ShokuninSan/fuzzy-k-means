package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import org.scalatest.{FlatSpec, Matchers}
import io.flatmap.ml.fuzzy.functions.initGaussian

class KMeansKernelSpec extends FlatSpec with Matchers {

  object TestKernel extends KMeansKernel

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))

  "calculateCentroids" should "return matrix with appropriate dimensions" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = TestKernel.calculateCentroids(data, u.t, 1)
    assert(v.cols == data.cols)
    assert(v.rows == u.cols)
  }

  "calculateCentroids" should "return calculate centers" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = TestKernel.calculateCentroids(data, u.t, 1)
    val expected = DenseMatrix(
      (2.0, 2.0)
    )
    assert(v == expected)
  }

  "updateMemberships" should "return matrix with appropriate dimensions 2" in {
    val c = 2
    val k = 3 // features
    val uInit = initGaussian(c, k)
    val d = DenseMatrix(
      (0.8, 0.6, 0.1, 0.4),
      (0.2, 0.4, 0.9, 0.6)
    )
    val u = TestKernel.updateMemberships(c, uInit, d, 2)
    println(u)
    assert(u.rows == d.cols)
    assert(u.cols == d.rows)
  }

}
