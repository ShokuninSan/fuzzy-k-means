package io.flatmap.ml.datasets

import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest.{Matchers, FlatSpec}

class IrisSpec extends FlatSpec with Matchers {

  "load" should "return data" in {
    val iris = Iris.load
    assert(iris.data.isInstanceOf[DenseMatrix[Double]])
    assert(iris.labels.isInstanceOf[DenseVector[Double]])
  }

}
