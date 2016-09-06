package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.clustering.kernels.{BreezeKMeansKernel, SparkKMeansKernel}
import io.flatmap.ml.fuzzy.numerics._
import io.flatmap.ml.normalization._
import io.flatmap.ml.test.util.TestSparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

class KMeansKernelSpec extends FlatSpec with Matchers with BeforeAndAfterEach with TestSparkContext  {

  val numClusters = 2
  val fuzziness = 2.5
  val epsilon = eps

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))

  "BreezeKMeansKernel.calculateCentroids" should "return matrix with appropriate dimensions" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = BreezeKMeansKernel.calculateCentroids(data, u.t, fuzziness)
    assert(v.cols == data.cols)
    assert(v.rows == u.cols)
  }

  "BreezeKMeansKernel.calculateCentroids" should "return calculate centers" in {
    val u = DenseMatrix.ones[Double](4, 1)
    val v = BreezeKMeansKernel.calculateCentroids(data, u.t, fuzziness)
    val expected = DenseMatrix(
      (2.0, 2.0)
    )
    assert(v == expected)
  }

  "SparkKMeansKernel.calculateCentroids" should "return calculated centers" in {
    // data matrix of shape (#points x #features)
    val data = new RowMatrix(sc.makeRDD(Seq(
      Vectors.dense(Array(1.0, 1.0)),
      Vectors.dense(Array(3.0, 3.0)),
      Vectors.dense(Array(1.0, 3.0)),
      Vectors.dense(Array(3.0, 1.0))
    )))
    // membership matrix of shape (#points x #centroids)
    val memberships = new RowMatrix(sc.makeRDD(Seq(
      Vectors.dense(Array(1.0)),
      Vectors.dense(Array(1.0)),
      Vectors.dense(Array(1.0)),
      Vectors.dense(Array(1.0))
    )))
    // centroid matrix of shape (#centroids x #features)
    val result = SparkKMeansKernel.calculateCentroids(data, memberships, fuzziness)
    val expected = new org.apache.spark.mllib.linalg.DenseMatrix(1, 2, Array(2.0, 2.0))
    assert(result == expected)
  }

  "BreezeKMeansKernel.calculateMemberships" should "return matrix with appropriate dimensions" in {
    val c = 2
    val k = 3 // features
    val d = DenseMatrix(
      (0.8, 0.6, 0.1, 0.4),
      (0.2, 0.4, 0.9, 0.6)
    )
    val u = BreezeKMeansKernel.calculateMemberships(d, fuzziness)
    assert(u.rows == d.rows)
    assert(u.cols == d.cols)
  }

  "SparkKMeansKernel.calculateMemberships" should "return matrix with appropriate dimensions" in {
    val d = new RowMatrix(sc.makeRDD(Seq(
      new DenseVector(Array(0.8, 0.6, 0.1, 0.4)),
      new DenseVector(Array(0.2, 0.4, 0.9, 0.6))
    )), 2,4)
    val u = SparkKMeansKernel.calculateMemberships(d, fuzziness)
    assert(u.numRows() == d.numRows())
    assert(u.numCols() == d.numCols())
  }

}
