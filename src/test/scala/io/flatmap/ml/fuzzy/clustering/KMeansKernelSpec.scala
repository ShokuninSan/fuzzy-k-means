package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.clustering.kernels.FuzzyKMeansKernel
import io.flatmap.ml.fuzzy.functions._
import io.flatmap.ml.normalization._
import org.scalatest.{FlatSpec, Matchers}

class KMeansKernelSpec extends FlatSpec with Matchers {

  val numClusters = 2
  val fuzziness = 2.5
  val epsilon = eps

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))

  "calculateCentroids" should "return matrix with appropriate dimensions" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = FuzzyKMeansKernel.calculateCentroids(data, u.t, fuzziness)
    assert(v.cols == data.cols)
    assert(v.rows == u.cols)
  }

  "calculateCentroids" should "return calculate centers" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = FuzzyKMeansKernel.calculateCentroids(data, u.t, fuzziness)
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
    val u = FuzzyKMeansKernel.calculateMemberships(d, fuzziness)
    assert(u.rows == d.rows)
    assert(u.cols == d.cols)
  }

}
