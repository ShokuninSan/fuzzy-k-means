package io.flatmap.ml.fuzzy.clustering

import breeze.linalg.DenseMatrix
import io.flatmap.ml.fuzzy.functions._

private[fuzzy] case class KMeansModel(centroids: DenseMatrix[Double]) {

  def predict(data: DenseMatrix[Double]): DenseMatrix[Double] = _predict(data)

  private def _predict(data: DenseMatrix[Double]): DenseMatrix[Double] = {
    data
  }

}

case class KMeans(c: Int, m: Int) extends KMeansKernel {

  def fit(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000): KMeansModel = {
    // step 1: c and m are already fixed. Initialize the partition matrix and r
    var u = initGaussian(c, data.cols)
    var r = 0

    while (r < maxIterations) {
      val u2 = u.copy

      // step 2: calculate cluster centers (eq 10.30)
      val um = pow(u, m).t
      val v = calculateCentroids(data, um, m)

      // step 3: update partition matrix
      val d = distance(v, data)
      u = updateMemberships(c, u2, d, m)

      // step 4: calculate the Frobenius norm of the two successive fuzzy partitions
      if (norm(u - u2) <= errorThreshold)
        r = maxIterations
      else
        r += 1
    }

    KMeansModel(u)
  }

}
