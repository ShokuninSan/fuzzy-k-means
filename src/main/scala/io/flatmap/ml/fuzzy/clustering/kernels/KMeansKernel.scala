package io.flatmap.ml.fuzzy.clustering.kernels

trait KMeansKernel[A, B, C] {

  def calculateCentroids(data: A, memberships: A, fuzziness: Double): B

  def calculateMemberships(distances: A, fuzziness: Double)(implicit normalizer: C): A

}
