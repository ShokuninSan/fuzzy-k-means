package io.flatmap.ml.fuzzy.normalization

import breeze.linalg.DenseMatrix
import io.flatmap.ml.normalization.{DenseMatrixMeanNormalizer, RowMatrixMeanNormalizer}
import io.flatmap.ml.test.util.TestSparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

class MeanNormalizerSpec extends FlatSpec with Matchers with BeforeAndAfterEach with TestSparkContext {

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))

  "normalize" should "return a matrix of same shape" in {
    val normalized = DenseMatrixMeanNormalizer.normalize(data)
    assert(normalized.rows == data.rows)
    assert(normalized.cols == data.cols)
  }

  "normalize" should "return a (column-wise) normalized matrix" in {
    val normalized = DenseMatrixMeanNormalizer.normalize(data)
    val expected = DenseMatrix(
      (0.125, 0.125),
      (0.375, 0.375),
      (0.125, 0.375),
      (0.375, 0.125)
    )
    assert(normalized == expected)
  }

  "normalize" should "return a (column-wise) normalized RowMatrix" in {
    val data = new RowMatrix(sc.makeRDD(Seq(
      new DenseVector(Array(1.0, 1.0)),
      new DenseVector(Array(3.0, 3.0)),
      new DenseVector(Array(1.0, 3.0)),
      new DenseVector(Array(3.0, 1.0)))),
      4, 2)
    val expected = new RowMatrix(sc.makeRDD(Seq(
      new DenseVector(Array(0.125, 0.125)),
      new DenseVector(Array(0.375, 0.375)),
      new DenseVector(Array(0.125, 0.375)),
      new DenseVector(Array(0.375, 0.125)))),
      4, 2)
    val result = RowMatrixMeanNormalizer.normalize(data)
    val expectedVector = expected.rows.collect().flatMap(_.toArray).toSeq
    val resultVector = result.rows.collect().flatMap(_.toArray).toSeq
    assert(resultVector == expectedVector)
  }

}
