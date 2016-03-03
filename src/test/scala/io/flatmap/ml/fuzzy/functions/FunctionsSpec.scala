package io.flatmap.ml.fuzzy.functions

import breeze.linalg.{eig, DenseMatrix}
import org.scalatest._

class FunctionsSpec extends FlatSpec with Matchers {

  "distance" should "calculate euclidean distance between two matrices" in {
    val X = DenseMatrix(
      (0.0, 2.0, 5.0, -1.0),
      (2.0, 4.0, 9.0, 4.0),
      (6.0, 6.0, 6.0, 0.0))
    val centroids = DenseMatrix(
      (1.0, 3.0, 7.5, 1.0),
      (5.0, 5.5, 6.0, 0.0)
    )
    val dist = DenseMatrix(
      (3.5, 6.264982043070834),
      (3.640054944640259, 6.020797289396148),
      (6.103277807866851, 1.118033988749895))
    assert(distance(X, centroids) == dist)
  }

  "distance" should "throw an error on invalid matrix dimensions" in {
    val a = DenseMatrix.zeros[Double](2, 3)
    val b = DenseMatrix.zeros[Double](3, 2)
    intercept[java.lang.AssertionError] { distance(a, b) }
  }

  "initGaussian" should "create a breeze matrix of random normal values" in {
    val rows = 5
    val cols = 3
    val m = initGaussian(n_samples = rows, n_features = cols)
    assert(m.rows == rows)
    assert(m.cols == cols)
  }

  "norm" should "create Frobenius norm of a matrix" in {
    val x = DenseMatrix(
      (0.0, 2.0, 5.0, -1.0),
      (2.0, 4.0, 9.0, 4.0),
      (6.0, 6.0, 6.0, 0.0))
    assert(norm(x) == 15.968719422671311)
  }

}
