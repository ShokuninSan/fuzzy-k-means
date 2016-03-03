package io.flatmap.ml.fuzzy

import breeze.linalg.{DenseMatrix}
import functions._

private[fuzzy] case class KMeansModel(centroids: DenseMatrix[Double]) {

  def predict(data: DenseMatrix[Double]): DenseMatrix[Double] = _predict(data)

  private def _predict(data: DenseMatrix[Double]): DenseMatrix[Double] = {
    data
  }

}

case class KMeans(c: Int, m: Int) {

  def fit(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000): KMeansModel = {
    // step 1: c and m are already fixed. Initialize the partition matrix and r
    val u = initGaussian(c, data.cols)
    var r = 0

    while (r < maxIterations) {
      val u2 = u.copy

      // step 2: calculate cluster centers

      // step 3: update partition matrix

      // step 4: calculate the Frobenius norm of the two successive fuzzy partitions
      if (norm(u - u2) <= errorThreshold)
        r = maxIterations
      else
        r += 1
    }

    KMeansModel(u)
  }

}
