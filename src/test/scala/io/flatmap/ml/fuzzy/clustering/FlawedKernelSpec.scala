package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.{DenseMatrix, DenseVector}
import io.flatmap.ml.fuzzy.clustering.kernels.{FuzzyKMeansKernel, KMeansKernel}
import io.flatmap.ml.fuzzy.functions._
import io.flatmap.ml.normalization.Normalizer
import org.scalatest.{FlatSpec, Matchers}

class FlawedKernelSpec extends FlatSpec with Matchers {

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

  object FlawedKMeansKernel extends KMeansKernel {

    def calculateCentroids(data: DenseMatrix[Double], memberships: DenseMatrix[Double], fuzziness: Double): DenseMatrix[Double] =
      FuzzyKMeansKernel.calculateCentroids(data, memberships, fuzziness)

    def calculateMemberships(distances: DenseMatrix[Double], fuzziness: Double)(implicit normalizer: Normalizer): DenseMatrix[Double] = {

      // THIS IS WRONG
      val _u = pow(distances, -2 / (fuzziness - 1))
      normalizer.normalize(_u)
    }

  }

  "KMeans.fit" should "solve the butterfly classification problem with large spatial distances" in {
    val model = KMeans(numClusters=2, fuzziness=2).fit(butterflyModel.mapValues(v => v * 10000))(FlawedKMeansKernel)

    // the central point is expected to have equal membership to both clusters
    val expectedMembership = DenseVector(0.5, 0.5)
    val membership = model.u(::, 0)

    val expectedCentroid0 = DenseVector(48500.0, 50000.0)
    val centroid0 = model.centroids(0,::).inner

    val expectedCentroid1 = DenseVector(91400.0, 50000.0)
    val centroid1 = model.centroids(1, ::).inner

    assert(closeTo(membership, expectedMembership))

    assert(
      (closeTo(centroid0, expectedCentroid0) && closeTo(centroid1, expectedCentroid1)) ||
        (closeTo(centroid0, expectedCentroid1) && closeTo(centroid1, expectedCentroid0))
    )
  }

}
