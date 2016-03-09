package io.flatmap.ml.datasets

import java.io.File
import java.nio.file.{StandardCopyOption, Paths, Files}
import breeze.linalg.{DenseVector, DenseMatrix}

private[datasets] case class IrisData(data: DenseMatrix[Double], labels: DenseVector[Double])

object Iris {

  private val PATH = "/iris.csv"

  private var irisData: Option[IrisData] = None

  def load: IrisData = irisData getOrElse {
    val inputStream = getClass.getResourceAsStream(PATH)
    val tmpFile = File.createTempFile(s"iris-${java.util.UUID.randomUUID()}-",".tmp")
    Files.copy(inputStream, Paths.get(tmpFile.getPath), StandardCopyOption.REPLACE_EXISTING)
    val data = breeze.linalg.csvread(tmpFile)
    irisData = Some(IrisData(
      data = data(::, 0 to 3),
      labels = data(::, -1).toDenseVector))
    tmpFile.delete
    irisData.get
  }

}
