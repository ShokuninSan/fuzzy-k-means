package io.flatmap.ml.fuzzy.functions

import breeze.linalg.{DenseVector, eig, DenseMatrix}
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

  "norm" should "create Frobenius norm of a matrix" in {
    val x = DenseMatrix(
      (0.0, 2.0, 5.0, -1.0),
      (2.0, 4.0, 9.0, 4.0),
      (6.0, 6.0, 6.0, 0.0))
    assert(norm(x) == 15.968719422671311)
  }

  "pow" should "power a breeze matrix" in {
    val x = DenseMatrix(
      (0.0, 2.0, 5.0, -1.0),
      (2.0, 4.0, 9.0, 4.0),
      (6.0, 6.0, 6.0, 0.0))
    val res = DenseMatrix(
      (0.0, 4.0, 25.0, 1.0),
      (4.0, 16.0, 81.0, 16.0),
      (36.0, 36.0, 36.0, 0.0))
    assert(pow(x, 2) == res)
  }

  "closeTo" should "evaluate correct closeness of vectors" in {
    val a = DenseVector(4.95, 5.0)
    val b = DenseVector(5.0, 5.0)

    // with default 'relDiff' of 1e-2 a difference of 0.05 should work (0.049 < 0.005)
    assert(closeTo(a, b, epsilon = 1e-2))

    // with a 'relDiff' of 0.005 we fail because '0.049 < 0.005' does not hold
    assertResult(false)(closeTo(a, b, epsilon = 1e-3))
  }

}
