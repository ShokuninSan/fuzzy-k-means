package io.flatmap.ml.normalization

import io.flatmap.ml.fuzzy.numerics._

trait Normalizer[T] {

  def normalize(m: T, epsilon: Double = eps): T

}
