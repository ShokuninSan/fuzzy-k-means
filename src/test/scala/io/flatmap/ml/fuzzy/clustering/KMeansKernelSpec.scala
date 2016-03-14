package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import org.scalatest.{FlatSpec, Matchers}
import io.flatmap.ml.fuzzy.functions.eps

class KMeansKernelSpec extends FlatSpec with Matchers {

  object TestKernel extends KMeansKernel {
    val numClusters = 2
    val fuzziness = 2.5
    val epsilon = eps
  }

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))

  "calculateCentroids" should "return matrix with appropriate dimensions" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = TestKernel.calculateCentroids(data, u.t)
    assert(v.cols == data.cols)
    assert(v.rows == u.cols)
  }

  "calculateCentroids" should "return calculate centers" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = TestKernel.calculateCentroids(data, u.t)
    val expected = DenseMatrix(
      (2.0, 2.0)
    )
    assert(v == expected)
  }

  "calculateMemberships" should "return matrix with appropriate dimensions" in {
    val c = 2
    val k = 3 // features
    val d = DenseMatrix(
      (0.8, 0.6, 0.1, 0.4),
      (0.2, 0.4, 0.9, 0.6)
    )
    val u = TestKernel.calculateMemberships(d)
    assert(u.rows == d.rows)
    assert(u.cols == d.cols)
  }

  "initMembershipMatrix" should "create a matrix with random uniformly distributed values" in {
    val numClusters = 3
    val numSamples = 5
    val u = TestKernel.initMembershipMatrix(numSamples, numClusters)
    assert(u.rows == numSamples)
    assert(u.cols == numClusters)
  }

  "normalize" should "return a matrix of same shape" in {
    val normalized = TestKernel.normalize(data)
    assert(normalized.rows == data.rows)
    assert(normalized.cols == data.cols)
  }

  "normalize" should "return a (column-wise) normalized matrix" in {
    val normalized = TestKernel.normalize(data)
    val expected = DenseMatrix(
      (0.125,0.125),
      (0.375,0.375),
      (0.125,0.375),
      (0.375, 0.125)
    )
    assert(normalized == expected)
  }

}
