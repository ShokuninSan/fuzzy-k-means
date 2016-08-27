import AssemblyKeys._

name := """fuzzy-k-means"""

organization := "io.flatmap"

version := "1.0.0-SNAPSHOT"

scalaVersion := "2.11.8"

val sparkVersion = "2.0.0"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "org.scalanlp" %% "breeze" % "0.12",
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
)

assemblySettings

parallelExecution in Test := false