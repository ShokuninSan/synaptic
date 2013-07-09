package perceptron

import scala.math._
import scala.util.Random

trait Neuron {

  val name: String
  val dendrites: List[Dendrite]

  // error ... indicated the error rate which should converge against 0 in each training epoch
  // bias  ... the threshold where the neuron fires
  protected var (error, bias, learningRate) = (0.0, (new Random).nextDouble * 2.0 - 1.0, 0.1)
  protected def input: Double

  var output: Double = 0.0
  def fire: Double
  def adjust: Unit

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
  def backpropagate(expected: Double): Unit
  def updateError(delta: Double): Unit

}

abstract class NeuronImpl(val name: String, inputLayer: List[Neuron]) extends Neuron with Activation {

  val dendrites = connect(inputLayer)

  def input = {
    error = 0.0 // error reset on new input
    dendrites.map(_.input).sum + bias;
  }

  def fire = { output = activate(input); output }

  def backpropagate(expected: Double) = updateError(expected - output)
 
  def updateError(delta: Double) {
    error += delta
    dendrites.foreach(_.updateError(delta))
  }

  def adjust = {
    val adjustment = error * derivativeFunction(output) * learningRate
    dendrites.foreach(_.adjust(adjustment))
    bias += adjustment
  }

  private def connect(ns: List[Neuron]): List[Dendrite] = ns.map(n => new Dendrite(n, (new Random).nextDouble * 2 * pow(ns.size, -0.5) - 1))

  override def toString = name + "[" + dendrites.mkString(",") + "]\n   "

}
