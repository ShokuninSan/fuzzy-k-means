package io.flatmap.ml.normalization

import breeze.linalg.{Axis, DenseMatrix, sum}
import io.flatmap.ml.fuzzy.functions._

trait Normalizer {

  def normalize(m: DenseMatrix[Double], epsilon: Double = eps): DenseMatrix[Double]

}
