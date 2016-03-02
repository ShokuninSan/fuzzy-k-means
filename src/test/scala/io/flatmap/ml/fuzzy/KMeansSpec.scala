package io.flatmap.ml.fuzzy

import breeze.linalg.DenseMatrix
import org.scalatest.{Matchers, FlatSpec}

class KMeansSpec extends FlatSpec with Matchers {

  "KMeans" should "instantiace a KMeans estimator" in {
    assert(KMeans(c=3, m=2).isInstanceOf[KMeans])
  }

  "KMeans.fit" should "return a KMeansModel" in {
    val data = DenseMatrix.zeros[Double](3,3)
    val model = KMeans(c=3, m=2).fit(data)
    assert(model.isInstanceOf[KMeansModel])
  }

}
