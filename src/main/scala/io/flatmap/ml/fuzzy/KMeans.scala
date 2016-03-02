package io.flatmap.ml.fuzzy

import breeze.linalg.DenseMatrix

private[fuzzy] case class KMeansModel(centroids: DenseMatrix[Double]) {

  def predict(data: DenseMatrix[Double]): DenseMatrix[Double] = ???

}

case class KMeans(c: Int, m: Int) {

  def fit(data: DenseMatrix[Double], errorThreshold: Double =  0.005, maxIterations: Int = 1000): KMeansModel = _fit(data)

  private def _fit(data: DenseMatrix[Double]): KMeansModel = {
    val centroids = DenseMatrix.zeros[Double](c, data.cols)
    KMeansModel(centroids)
  }

}
