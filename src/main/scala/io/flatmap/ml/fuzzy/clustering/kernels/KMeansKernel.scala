package io.flatmap.ml.fuzzy.clustering.kernels

import breeze.linalg.DenseMatrix
import io.flatmap.ml.normalization.Normalizer

trait KMeansKernel {

  def calculateCentroids(data: DenseMatrix[Double], memberships: DenseMatrix[Double], fuzziness: Double): DenseMatrix[Double]

  def calculateMemberships(distances: DenseMatrix[Double], fuzziness: Double)(implicit normalizer: Normalizer): DenseMatrix[Double]

}
