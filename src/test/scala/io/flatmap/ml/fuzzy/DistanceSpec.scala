package io.flatmap.ml.fuzzy

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest._

class DistanceSpec extends FlatSpec with Matchers {

  "distance" should "calculate euclidean distance between two matrices" in {
    val X = DenseMatrix(
      (0.0, 2.0, 5.0, -1.0),
      (2.0, 4.0, 9.0, 4.0),
      (6.0, 6.0, 6.0, 0.0))
    val centroids = DenseMatrix(
      (1.0, 3.0, 7.5, 1.0),
      (5.0, 5.5, 6.0, 0.0)
    )
    print(distance(X, centroids))
  }

  "distance" should "throw an error on invalid matrix dimensions" in {
    val a = DenseMatrix.zeros[Double](2, 3)
    val b = DenseMatrix.zeros[Double](3, 2)
    intercept[java.lang.AssertionError] { distance(a, b) }
  }

  "euclideanDistance" should "be correct with dense vectors" in {
    val v1 = DenseVector(1.0, 1.0)
    val v2 = DenseVector(9.0, -9.0)
    assert(euclideanDistance(v1, v2) === 12.806248474865697)
  }

}
