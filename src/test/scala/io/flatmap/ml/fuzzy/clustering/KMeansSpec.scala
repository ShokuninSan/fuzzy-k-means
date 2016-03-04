package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import org.scalatest.{FlatSpec, Matchers}

class KMeansSpec extends FlatSpec with Matchers {

  "KMeans" should "instantiace a KMeans estimator" in {
    assert(KMeans(c=3, m=2).isInstanceOf[KMeans])
  }

  "KMeans.fit" should "return a KMeansModel" in {
    val data = DenseMatrix.zeros[Double](3,3)
    val model = KMeans(c=3, m=2).fit(data)
    assert(model.isInstanceOf[KMeansModel])
  }

  "KMeans.fit" should "return a KMeansModel with correct centroids and memberships" in {
    val data = DenseMatrix(
      (1.0, 1.0),
      (3.0, 3.0),
      (1.0, 3.0),
      (3.0, 1.0))
    val model = KMeans(c=1, m=2).fit(data)
    val expectedMemberships = DenseMatrix((1.0, 1.0, 1.0, 1.0))
    val expectedCentroids = DenseMatrix((2.0, 2.0))
    assert(model.u == expectedMemberships)
    assert(model.centroids == expectedCentroids)
  }

  "KMeans.fit" should "return a KMeansModel with correct centroids" in {
    val data = DenseMatrix(
      (1.0, 1.0, 1.0),
      (3.0, 3.0, 3.0))
    val model = KMeans(c=1, m=2).fit(data)
    val expectedCentroids = DenseMatrix((2.0, 2.0, 2.0))
    assert(model.centroids == expectedCentroids)
  }

  "KMeans.predict" should "return a membership matrix" in {
    val data = DenseMatrix.zeros[Double](3,3)
    val model = KMeans(c=3, m=2).fit(data)
    val memberships = model.predict(data)
    assert(memberships.isInstanceOf[DenseMatrix[Double]])
  }

}
