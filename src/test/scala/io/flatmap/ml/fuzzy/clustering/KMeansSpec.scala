package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest.{FlatSpec, Matchers}
import io.flatmap.ml.fuzzy.functions.closeTo

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

  "KMeans.fit" should "solve the butterfly classification problem" in {
    val data = DenseMatrix(
      (7.0, 5.0), // the central point
      (6.0, 5.0),
      (5.0, 4.0),
      (5.0, 5.0),
      (5.0, 6.0),
      (4.0, 3.0),
      (4.0, 5.0),
      (4.0, 7.0),
      (8.0, 5.0),
      (9.0, 4.0),
      (9.0, 5.0),
      (9.0, 6.0),
      (10.0, 3.0),
      (10.0, 5.0),
      (10.0, 7.0)
    )

    val model = KMeans(c=2, m=2).fit(data)

    // the central point is expected to have equal membership to both clusters
    val expectedMembership = DenseVector(0.5, 0.5)
    val membership = model.u(::, 0)

    val expectedCentroid0 = DenseVector(4.85, 5.0)
    val centroid0 = model.centroids(0,::).inner

    val expectedCentroid1 = DenseVector(9.14, 5.0)
    val centroid1 = model.centroids(1, ::).inner

    assert(closeTo(membership, expectedMembership))

    assert(
      (closeTo(centroid0, expectedCentroid0) && closeTo(centroid1, expectedCentroid1)) ||
      (closeTo(centroid0, expectedCentroid1) && closeTo(centroid1, expectedCentroid0))
    )

  }

  "KMeans.predict" should "return a membership matrix" in {
    val data = DenseMatrix.zeros[Double](3,3)
    val model = KMeans(c=3, m=2).fit(data)
    val memberships = model.predict(data)
    assert(memberships.isInstanceOf[DenseMatrix[Double]])
  }

}
