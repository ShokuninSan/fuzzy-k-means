package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import org.scalatest.{FlatSpec, Matchers}

class KMeansKernelSpec extends FlatSpec with Matchers {

  object TestKernel extends KMeansKernel

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))
  val u = DenseMatrix.ones[Double](4, 1) // dim: n_samples * n_clusters

  "calculateCentroids" should "return matrix with appropriate dimensions" in {
    val v = TestKernel.calculateCentroids(data, u.t, 1)
    assert(v.cols == data.cols)
    assert(v.rows == u.cols)
  }

  "calculateCentroids" should "return calculate centers" in {
    val v = TestKernel.calculateCentroids(data, u.t, 1)
    val expected = DenseMatrix(
      (2.0, 2.0)
    )
    assert(v == expected)
  }

}
