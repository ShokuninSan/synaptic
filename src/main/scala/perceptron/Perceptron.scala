package perceptron

import scala.util.Random

class Perceptron(layout: List[Int], rnd: Random) {
  val layers = build(layout, rnd)
 
  def output(ins: List[Double]) = {
    layers.head.zip(ins).foreach { case (n, in) => n.out = in }
    layers.tail.foldLeft(ins) { (z, l) => l.map(_.output) }
  }
 
  def train(ins: List[Double], outs: List[Double]) = {
    val outputs = output(ins)
    layers.last.zip(0 until outs.length).foreach {case (n, m) => n.expectation(outs(m))}
    layers.foreach(_.foreach(_.adjust))
  }
 
  override def toString = layers.mkString("\n")
 
  private def build(layout: List[Int], rnd: Random) =
    layout.zip(1 to layout.size).foldLeft(List(List[Neuron]())) {
      case (z, (n, l)) => buildLayer("L"+l, n, z.head, rnd) :: z
    }.reverse.tail
 
 
  private def buildLayer(name: String, n: Int, lower: List[Neuron], rnd: Random) =
    (0 until n) map { n => new Neuron(name+"N"+n, lower, rnd) } toList
}
