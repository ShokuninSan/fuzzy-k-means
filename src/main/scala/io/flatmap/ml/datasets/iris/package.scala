package io.flatmap.ml.datasets

import java.io.File
import breeze.linalg.{DenseVector, DenseMatrix}

package object iris {

  private val PATH = "/iris.csv"

  def load: (DenseMatrix[Double], DenseVector[Double]) = {
    val url = getClass.getResource(PATH)
    val data = breeze.linalg.csvread(new File(url.toURI))
    (data(::, 0 to 3), data(::, -1).toDenseVector)
  }

}
