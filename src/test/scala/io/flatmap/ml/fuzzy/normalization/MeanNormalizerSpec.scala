package io.flatmap.ml.fuzzy.normalization

import breeze.linalg.DenseMatrix
import io.flatmap.ml.normalization.MeanNormalizer
import org.scalatest.{FlatSpec, Matchers}

class MeanNormalizerSpec extends FlatSpec with Matchers {

  val data = DenseMatrix(
    (1.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (3.0, 1.0))

  "normalize" should "return a matrix of same shape" in {
    val normalized = MeanNormalizer.normalize(data)
    assert(normalized.rows == data.rows)
    assert(normalized.cols == data.cols)
  }

  "normalize" should "return a (column-wise) normalized matrix" in {
    val normalized = MeanNormalizer.normalize(data)
    val expected = DenseMatrix(
      (0.125,0.125),
      (0.375,0.375),
      (0.125,0.375),
      (0.375, 0.125)
    )
    assert(normalized == expected)
  }

}
