package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.functions._
import org.scalatest.{FlatSpec, Matchers}

class IterativeOptimizationSpec extends FlatSpec with Matchers {

  object TestKernel extends IterativeOptimization {
    val numClusters = 2
    val fuzziness = 2.5
    val epsilon = eps
  }

  "initMembershipMatrix" should "create a matrix with random uniformly distributed values" in {
    val numClusters = 3
    val numSamples = 5
    val u = TestKernel.initMembershipMatrix(numSamples, numClusters)
    assert(u.rows == numSamples)
    assert(u.cols == numClusters)
  }

  "product of a unit matrix of shape (n x 1) and a sum matrix of shape (1 x m)" should "produce a (n x m) matrix with n copies of the sums" in {
    val n = 3
    val m = 5
    val ones = allOnesMatrix(rows = n, cols = 1)
    val values = DenseMatrix.create(rows = 1, cols = m, Array.tabulate[Double](m)(i => (i + 1) * 2))

    val product = ones * values

    val expected = DenseMatrix(
      (2.0, 4.0, 6.0, 8.0, 10.0),
      (2.0, 4.0, 6.0, 8.0, 10.0),
      (2.0, 4.0, 6.0, 8.0, 10.0)
    )

    assert(product == expected)
  }

}
