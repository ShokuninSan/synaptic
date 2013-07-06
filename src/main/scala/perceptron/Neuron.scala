package perceptron

import scala.math._
import scala.util.Random

abstract class Neuron(val name: String, inputLayer: List[Neuron]) extends Activation {

  val dendrites = connect(inputLayer)

  def input = {
    error = 0.0 // error reset on new input
    dendrites.map(_.input).sum + bias;
  }

  /**
   * This function is called on the output neuron within the training epochs.
   *
   * Once this function is called on the output neuron, the whole networks neurons
   * will be updated recursively, i.e. their 'error' field gets updated by a delta
   * (which is calculated on the dendrites &rarr; delta * weight.
   *
   * Both, the expectation and {Neuron, Dentrite}.updateError functions form the
   * backpropagation-rule.
   *
   * @param expected value for output neuron
   */
  def backpropagate(expected: Double) = updateError(expected - out)
 
  def updateError(delta: Double) {
    error += delta
    dendrites.foreach(_.updateError(delta))
  }

  private def connect(ns: List[Neuron]): List[Dendrite] =
    ns.map(n => new Dendrite(n, (new Random).nextDouble * 2 * pow(ns.size, -0.5) - 1))

  override def toString = name + "[" + dendrites.mkString(",") + "]\n   "
}
