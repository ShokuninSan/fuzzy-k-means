package io.flatmap.ml.datasets

import breeze.linalg.{DenseVector, DenseMatrix}
import org.scalatest.{Matchers, FlatSpec}

class IrisSpec extends FlatSpec with Matchers {

  "load" should "return data" in {
    val (data, labels) = iris.load
    assert(data.isInstanceOf[DenseMatrix[Double]])
    assert(labels.isInstanceOf[DenseVector[Double]])
  }

}
