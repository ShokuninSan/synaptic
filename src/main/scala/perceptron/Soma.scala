package perceptron

import scala.concurrent.Future

trait Soma {

  val name: String

  val dendrites: List[Dendrite]

  /**
   * The indispensable attributes of a Neuron which implements this trait.
   *
   * error .......... the delta of the expected output and the actual output
   * bias  .......... alternatively the threshold Theta can be implemented by addition of a input Neuron, a so-called
   *                  on-Neuron or Bias (translated in german as Tendenz, Neigung, Ausrichtung). See also:
   *                  <a href="http://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks">
   *                    Role of Bias in Neural Networks @ StackOverflow
   *                  </a>
   * learningRate ... the learning rate Eta
   */
  var (error, bias, learningRate) = (0.0, Util.randomBiasWeight, 0.1)

  def input: Double

  var output: Double = 0.0

  def fire: Future[Double]

  def applyDeltaRule: Future[Double]

  def updateError(delta: Double): Future[Double]

}