name := """fuzzy-k-means"""

organization := "io.flatmap"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "org.scalanlp" %% "breeze" % "0.12"
)
