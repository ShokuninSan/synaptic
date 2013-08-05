package perceptron

import scala.util.Random
import scala.concurrent.Future

trait Soma {

  val name: String

  val dendrites: List[Dendrite]

  // error ... indicated the error rate which should converge against 0 in each training epoch
  // bias  ... the threshold where the neuron fires
  var (error, bias, learningRate) = (0.0, (new Random).nextDouble * 2.0 - 1.0, 0.1)

  def input: Double

  var output: Double = 0.0

  def fire: Future[Double]

  def adjust: Future[Double]

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
  def backPropagate(expected: Double): Future[Double]

  def updateError(delta: Double): Future[Double]

}