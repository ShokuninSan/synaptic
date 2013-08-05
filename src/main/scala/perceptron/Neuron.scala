package perceptron

import scala.math._
import scala.util.Random
import ActivationFunctions._
import scala.concurrent.{ExecutionContext, Future}
import ExecutionContext.Implicits.global

abstract class Neuron(val name: String, inputLayer: List[Neuron]) extends Soma with Activation {

  val dendrites = connect(inputLayer)

  override def input: Double = {
    error = 0.0 // error reset on new input
    dendrites.map(_.input).sum + bias
  }

  def feed(input: Double) = output = input

  override def fire = Future {
    output = activate(input)
    output
  }

  def out: Double = output

  override def backPropagate(expected: Double): Future[Double] = updateError(expected - output)
 
  override def updateError(delta: Double): Future[Double] = Future {
    error += delta
    dendrites.foreach(_.updateError(delta))
    error
  }

  override def adjust: Future[Double] = Future {
    val adjustment = error * derivativeFunction(output) * learningRate
    dendrites.foreach(_.adjust(adjustment))
    bias += adjustment
    bias
  }

  private def connect(ns: List[Neuron]): List[Dendrite] = ns.map { n =>
    new Dendrite(n, (new Random).nextDouble * 2 * pow(ns.size, -0.5) - 1)
  }

  override def toString = name + "[" + dendrites.mkString(",") + "]\n   "

}

object Neuron {

  def apply(name: String, lower: List[Neuron], activation: ActivationFunctions.Value): Neuron = activation match {
    case HyperbolicTangent => new Neuron(name, lower) with HyperbolicTangent
    case Sigmoid => new Neuron(name, lower) with Sigmoid
  }

}