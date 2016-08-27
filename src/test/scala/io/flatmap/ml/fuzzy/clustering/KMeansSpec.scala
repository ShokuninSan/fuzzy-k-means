package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest.{FlatSpec, Matchers}
import io.flatmap.ml.fuzzy.numerics.closeTo

class KMeansSpec extends FlatSpec with Matchers {

  val butterflyModel = DenseMatrix(
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

  "KMeans" should "instantiace a KMeans estimator" in {
    assert(KMeans(numClusters=3, fuzziness=2).isInstanceOf[KMeans])
  }

  "KMeans.fit" should "return a KMeansModel" in {
    val data = DenseMatrix.zeros[Double](3,3)
    val model = KMeans(numClusters=3, fuzziness=2).fit(data)
    assert(model.isInstanceOf[KMeansModel])
  }

  "KMeans.fit" should "return a KMeansModel with correct centroids and memberships" in {
    val data = DenseMatrix(
      (1.0, 1.0),
      (3.0, 3.0),
      (1.0, 3.0),
      (3.0, 1.0))
    val model = KMeans(numClusters=1, fuzziness=2).fit(data)
    val expectedMemberships = DenseMatrix((1.0, 1.0, 1.0, 1.0))
    val expectedCentroids = DenseMatrix((2.0, 2.0))
    assert(model.u == expectedMemberships)
    assert(model.centroids == expectedCentroids)
  }

  "KMeans.fit" should "return a KMeansModel with correct centroids" in {
    val data = DenseMatrix(
      (1.0, 1.0, 1.0),
      (3.0, 3.0, 3.0))
    val model = KMeans(numClusters=1, fuzziness=2).fit(data)
    val expectedCentroids = DenseMatrix((2.0, 2.0, 2.0))
    assert(model.centroids == expectedCentroids)
  }

  "KMeans.fit" should "solve the butterfly classification problem" in {
    val model = KMeans(numClusters=2, fuzziness=2).fit(butterflyModel)

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

  "KMeans.fit" should "solve the butterfly classification problem with large spatial distances" in {
    val model = KMeans(numClusters=2, fuzziness=2).fit(butterflyModel.mapValues(v => v * 1000))

    // the central point is expected to have equal membership to both clusters
    val expectedMembership = DenseVector(0.5, 0.5)
    val membership = model.u(::, 0)

    val expectedCentroid0 = DenseVector(4850.0, 5000.0)
    val centroid0 = model.centroids(0,::).inner

    val expectedCentroid1 = DenseVector(9140.0, 5000.0)
    val centroid1 = model.centroids(1, ::).inner

    assert(closeTo(membership, expectedMembership))

    assert(
      (closeTo(centroid0, expectedCentroid0) && closeTo(centroid1, expectedCentroid1)) ||
        (closeTo(centroid0, expectedCentroid1) && closeTo(centroid1, expectedCentroid0))
    )
  }

  "KMeans.predict" should "return a membership matrix" in {
    val data = DenseMatrix.zeros[Double](3,3)
    val model = KMeans(numClusters=3, fuzziness=2).fit(data)
    val memberships = model.predict(data)
    assert(memberships.isInstanceOf[DenseMatrix[Double]])
  }

  "KMeans.predict" should "return correct membership values" in {
    val model = KMeans(numClusters=2, fuzziness=2.1).fit(butterflyModel)
    val newData = DenseMatrix(
      (7.0, 3.0)
    )
    val memberships = model.predict(newData)
    assert(closeTo(memberships(::, 0), DenseVector(0.5, 0.5), 1e-2))
  }

}
