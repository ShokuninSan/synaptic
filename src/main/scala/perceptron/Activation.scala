package perceptron

import scala.math._
import scala.util.Random

object ActivationFunctions extends Enumeration {
  type ActivationFunctions = Value
  val HyperbolicTangent, Sigmoid = Value
}

trait Activation {

  val dendrites: List[Dendrite]

  // out   ... the output value of the neuron
  // error ... indicated the error rate which should converge against 0 in each training epoch
  // bias  ... the threshold where the neuron fires
  var (out, error, bias, learningRate) = (0.0, 0.0, (new Random).nextDouble * 2.0 - 1.0, 0.1)

  def input: Double
  def output: Double
  def adjust: Unit
  def derivative(out: Double): Double
  def activate(x: Double): Double
}

trait HyperbolicTangent extends Activation {

  val (a, b) = (1.7159, 2.0/3.0)

  def output = {
    out = a * activate(b*input)
    out
  }

  def derivative(out: Double) = a * b * (1-pow(activate(b*out), 2))

  def activate(x: Double) = {
    val ex = exp(2.0*x)
    (ex - 1) / (ex + 1)
  }
}

trait Sigmoid extends Activation {

  def output = {
    out = activate(input)
    out
  }

  def derivative(out: Double) = out * (1. - out)

  def activate(x: Double) = 1. / (1. + exp (-x))
}