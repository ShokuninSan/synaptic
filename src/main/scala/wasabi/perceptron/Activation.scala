package wasabi.perceptron

import scala.math._

object ActivationFunctions extends Enumeration {

  type ActivationFunctions = Value
  val HyperbolicTangent, Sigmoid = Value

}

trait Activation {

  def activate(input: Double): Double

  def derivativeFunction(out: Double): Double

  def activationFunction(x: Double): Double

}

trait HyperbolicTangent extends Activation {

  val (a, b) = (1.7159, 2.0/3.0)

  def activate(input: Double): Double = a * activationFunction(b*input)

  def derivativeFunction(out: Double) = a * b * (1-pow(activationFunction(b*out), 2))

  def activationFunction(x: Double) = {
    val ex = exp(2.0*x)
    (ex - 1) / (ex + 1)
  }

}

trait Sigmoid extends Activation {

  def activate(input: Double): Double = activationFunction(input)

  def derivativeFunction(out: Double) = out * (1. - out)

  def activationFunction(x: Double) = 1. / (1. + exp (-1 * x))

}