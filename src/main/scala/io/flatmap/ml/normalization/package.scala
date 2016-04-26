package io.flatmap.ml

import breeze.linalg.{Axis, DenseMatrix, sum}
import io.flatmap.ml.fuzzy.functions._

package object normalization {

  implicit object MeanNormalizer extends Normalizer {

    /**
      * Normalizes the values of a matrix
      *
      * This method normalizes each value over the sum of the respective column. Before the matrix is returned, NaN (Not a Number) values are
      * substituted with epsilon.
      *
      * @param m Matrix (e.g. memberships or distances)
      * @return Normalized matrix
      */
    def normalize(m: DenseMatrix[Double], epsilon: Double = eps): DenseMatrix[Double] = {
      val ones: DenseMatrix[Double] = allOnesMatrix(m.rows, 1)
      val sumByColumns: DenseMatrix[Double] = sum(m, Axis._0).inner.toDenseMatrix // shape (1 x #columns)
      val normalized: DenseMatrix[Double] = m / (ones * sumByColumns) // '/' calculates element-wise, i.e. (#rows x #cols) / (#rows x #cols)
      fmax(normalized, epsilon) // substitute NaNs by epsilon
    }

  }

}
