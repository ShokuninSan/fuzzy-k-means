package io.flatmap.ml.fuzzy.clustering

import breeze.linalg._
import io.flatmap.ml.fuzzy.numerics._
import io.flatmap.ml.normalization.Normalizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix

package object kernels {

  implicit object FuzzyKMeansKernel extends KMeansKernel {

    /**
      * Calculate cluster centroids
      *
      * This is a vectorized implementation according to Timothy Ross book "Fuzzy Logic with Engineering Applications", p. 352,
      * equation (10.30)
      *
      * @param data Matrix of samples of shape (#datapoints x #features)
      * @param memberships Matrix of membership degrees of data points to clusters of shape (#centroids x #datapoints)
      * @return Matrix of shape #clusters x #features
      */
    def calculateCentroids(data: DenseMatrix[Double], memberships: DenseMatrix[Double], fuzziness: Double): DenseMatrix[Double] = {
      val _u = pow(memberships, fuzziness)
      val ones = allOnesMatrix(data.rows, data.cols)
      (_u * data) / (_u * ones)
    }

    /**
      * Calculate cluster centroids
      *
      * This is a scalable implementation according to Timothy Ross book "Fuzzy Logic with Engineering Applications", p. 352,
      * equation (10.30)
      *
      * @param data Matrix of samples of shape (#datapoints x #features)
      * @param memberships Matrix of membership degrees of data points to clusters of shape (#datapoints x #centroids)
      * @return Matrix of shape #clusters x #features
      */
    def calculateCentroids(data: RowMatrix, memberships: RowMatrix, fuzziness: Double): DenseMatrix[Double] = {
      val f = (v: Vector) => DenseVector.apply(v.toArray)
      val _m = pow(memberships, fuzziness).rows.map(f).cache()
      val _d = data.rows.map(f)
      val dotProduct = dot(_m, _d)
      val margin = _m.reduce(_ :+ _)
      dotProduct(::, *).map(_ :/ margin)
    }

    /**
      * Calculate cluster membership of data points
      *
      * This is a vectorized implementation according to Timothy Ross book "Fuzzy Logic with Engineering Applications", p. 353,
      * equation (10.32a)
      *
      * @param distances Matrix of distances between centroids and data points of shape (#datapoints x #centroids)
      * @return Matrix of calculated membership degrees of shape (#clusters x #datapoints)
      */
    def calculateMemberships(distances: DenseMatrix[Double], fuzziness: Double)(implicit normalizer: Normalizer): DenseMatrix[Double] = {
      val _u = pow(pow(distances, 2 / (fuzziness - 1)), -1)
      normalizer.normalize(_u)
    }

  }

}
