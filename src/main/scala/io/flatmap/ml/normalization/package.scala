package io.flatmap.ml

import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}
import io.flatmap.ml.fuzzy.numerics._
import org.apache.spark.mllib.linalg.distributed.RowMatrix

package object normalization {

  implicit object DenseMatrixMeanNormalizer extends Normalizer[DenseMatrix[Double]] {

    /**
      * Normalizes the values of a matrix
      *
      * This method normalizes each value over the sum of the respective column. Before the matrix is returned, NaN (Not a Number) values are
      * substituted with epsilon.
      *
      * @param m Matrix (e.g. memberships or distances)
      * @return Normalized matrix
      */
    override def normalize(m: DenseMatrix[Double], epsilon: Double = eps): DenseMatrix[Double] = {
      val ones: DenseMatrix[Double] = allOnesMatrix(m.rows, 1)
      val sumByColumns: DenseMatrix[Double] = sum(m, Axis._0).inner.toDenseMatrix // shape (1 x #columns)
      val normalized: DenseMatrix[Double] = m / (ones * sumByColumns) // '/' calculates element-wise, i.e. (#rows x #cols) / (#rows x #cols)
      fmax(normalized, epsilon) // substitute NaNs by epsilon
    }

  }

  implicit object RowMatrixMeanNormalizer extends Normalizer[RowMatrix] {

    override def normalize(m: RowMatrix, epsilon: Double = eps): RowMatrix = {
      val denseVectorRDD = m.rows.map(v => DenseVector[Double](v.toArray)).cache()
      val sumByColumns = denseVectorRDD.reduce((a, b) => a :+ b)
      new RowMatrix(
        rows = denseVectorRDD.map(_ :/ sumByColumns).map {
          dv =>
            val fixed = dv.map(double => if(double.isNaN) epsilon else double)
            org.apache.spark.mllib.linalg.Vectors.dense(fixed.toArray)
        },
        nRows = m.numRows(),
        nCols = m.numCols().toInt
      )
    }

  }

}
