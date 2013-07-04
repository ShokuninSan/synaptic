package perceptron

import scala.math._
import scala.util.Random

class Neuron(nm: String, ns: List[Neuron], rnd: Random) {
  val (a, b, rate) = (1.7159, 2.0/3.0, 0.1)
  val dendrites = connect(ns)
  val name = nm
 
  // need to remember output and gather error for training
  var (out, error, bias) = (0.0, 0.0, rnd.nextDouble * 2.0 - 1.0)
 
  def input = {
    error = 0.0 // error reset on new input
    dendrites.map(_.input).sum + bias;
  }
 
  def output = {
    out = a * tanh(b*input)
    out
  }
 
  def expectation(expected: Double) = updateError(expected - out)
 
  def updateError(delta: Double) {
    error += delta
    dendrites.foreach(_.updateError(delta))
  }
 
  def adjust {
    val adjustment = error * deriv(out) * rate
    dendrites.foreach(_.adjust(adjustment))
    bias += adjustment
  }
 
  override def toString = name + "[" + dendrites.mkString(",") + "]\n   "
 
  // Derivative of our output function
  private def deriv(out: Double) = a * b * (1-pow(tanh(b*out), 2))
 
  private def connect(ns: List[Neuron]): List[Dendrite] =
    ns.map(n => new Dendrite(n, rnd.nextDouble * 2 * pow(ns.size, -0.5) - 1))
 
  // Hyperbolic tangent function
  private def tanh(x: Double) = {
    val ex = exp(2.0*x)
    (ex - 1) / (ex + 1)
  }
}
