package io.flatmap.ml.fuzzy.clustering.example

import breeze.linalg._
import io.flatmap.ml.datasets._
import io.flatmap.ml.fuzzy.clustering.KMeans
import java.io.File

object IrisApplication {

  def main(args: Array[String]): Unit = {

    // load the Iris dataset and train the model
    val iris = Iris.load
    val model = KMeans(numClusters = 3, fuzziness = 2).fit(iris.data)

    val soft = model.u.t
    val hard = breeze.linalg.argmax(model.u(::, *)).inner.mapValues(_.toDouble).toDenseMatrix.t
    val union = DenseMatrix.horzcat(
      iris.data,
      soft,
      hard
    )
    
    csvwrite(new File("iris_predictions.csv"), union)
  }

}
