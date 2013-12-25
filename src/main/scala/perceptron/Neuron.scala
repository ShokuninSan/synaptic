package perceptron

import scala.math._
import scala.util.Random
import ActivationFunctions._
import scala.concurrent.{ExecutionContext, Future}
import ExecutionContext.Implicits.global

abstract class Neuron(val name: String, inputLayer: List[Neuron]) extends Soma with Activation {

  val dendrites = connect(inputLayer)

  /**
   * This function calculates and finally returns the input for this Neuron.
   *
   * The input value for this Neuron (if it's not an Input Neuron per se) is calculated by `map`ing over the dendrites
   * of this Neuron and executing their [[perceptron.Dendrite.input]] functions, which in turn just multiply the feeding
   * input Neurons' outputs by the Dendrite's weight. This algorithm is also known as:
   *
   *   netj = sum(oi*wij + ... + on*wnj)
   *
   * @return
   */
  override def input: Double = {
    error = 0.0
    dendrites.map(_.input).sum + bias
  }

  /**
   * Execute this function if the Neuron is an input Neuron.
   *
   * If this Neuron acts as a input Neuron to the Network, pass the desired input value
   * via [[perceptron.Neuron.feed()]]. As a result the Neuron's output value will be exactly
   * the given value without any modification.
   *
   * @param input Input value for the given Network.
   */
  def feed(input: Double) = output = input

  /**
   * Forward propagation of input.
   *
   * This function calls the corresponding activation Function with the calculated [[perceptron.Neuron.input]].
   * The mentioned input is calculated by execution of [[perceptron.Dendrite.input]] of all feeding
   * [[perceptron.Dendrite]]s where the algorithm is as easy as the factor of the corresponding input Neuron's output
   * multiplied by the weight:
   *
   *   netj = sum(oi*wij + ... + on*wnj)
   *   sigmoidActivation(netj)
   *
   * @return Future[Double] the output of the Neuron
   */
  override def fire = Future {
    output = activate(input)
    output
  }

  /**
   * Returns the result of the last forward propagations result.
   *
   * Forward propagation is done by [[perceptron.Neuron.fire]] and the result is stored in the var `output` which is
   * inherited from the [[perceptron.Soma]] trait. A query on this member returns this calculated result.
   *
   * @return
   */
  def out: Double = output

  /**
   * This function is called when the Neuron is an output Neuron.
   *
   * This function is the counterpart of the [[perceptron.Neuron.feed()]] function. It should be called only on output
   * Neurons. This function initiates back propagation for it's path by calling it's own [[perceptron.Neuron.updateError()]]
   * function.
   *
   * The passed value is the expected output which is given by corresponding training sample (pattern).
   *
   * @param expected value for output neuron
   * @return the updated delta/error of this Neuron
   */
  override def backPropagate(expected: Double): Future[Double] = updateError(expected - output)

  /**
   * Prepare Neuron for backpropagation.
   *
   * Step 1 of backpropagation. This function calculates the delta as part of the backpropagation algorithm:
   *
   *   f'(netj) * sumk(deltak * wjk)
   *
   * So this function calculates the `deltak * wjk` part for 'hidden' Neurons respectively `tj - oj` for output Neurons.
   * More precisely, these deltas are not calculated within this function, moreover these values are already passed as
   * `delta` parameter. See implementation of [[perceptron.Dendrite.updateError()]].
   *
   * @param delta calculated delta. Either `deltak * wjk` for hidden Neurons or `tj - oj` for output Neurons.
   * @return the updated delta/error.
   */
  override def updateError(delta: Double): Future[Double] = Future {
    error += delta
    dendrites.foreach(_.updateError(delta))
    error
  }

  /**
   * This function implements the computation of the deltaj as part of the backpropagation rule:
   *
   *   f'(netj) (tj-oj)           ... in case of an output Neuron, or
   *   f'(netj) sum(deltak * wjk) ... in case of a hidden Neuron
   *
   * The return value of this function gets applied to
   *
   * 1) the computation of weigths of each input Neuron: see [[perceptron.Neuron.adjust]] and [[perceptron.Dendrite.adjust]]
   * 2) the computation of the Bias (on-Neuron): see [[perceptron.Neuron.adjust]]
   *
   * See also book Simulation Neuronaler Netze, A. Zell (2000), ch. 5.9.4 Backpropagation-Regel on page 86, figure 5.31.
   *
   * @return The deltaj
   */
  def deltaj = derivativeFunction(output) * error

  /**
   * Completes the backpropagation process by applying the `deltawij` to the weights of the input Neurons as well as to
   * the Bias (on-Neuron).
   *
   * This function computes:
   *
   * 1) the deltawij as described in Simulation Neuronaler Netze, A. Zell (2000), ch. 5.9.4 Backpropagation-Regel on
   *    page 86, figure 5.30
   * 2) the adjustment for the Bias which is added to the sum of the input weights within the next forward propagation
   * cycle.
   *
   * See also:
   *   <a href="http://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks">
   *     Role of Bias in Neural Networks @ StackOverflow
   *   </a>
   */
  override def adjust: Future[Double] = Future {
    val adjustment = learningRate * deltaj
    dendrites.foreach(_.adjust(adjustment))
    bias += adjustment
    bias
  }

  /**
   * This function is executed upon instantiation of a Neuron.
   *
   * [[perceptron.Neuron.connect()]] connects (as the name implies) this Neuron via [[perceptron.Dendrite]]s to the
   * corrensponding input layer, i.e. either input Neurons or Neurons of a hidden layer. The Dendrites are created
   * with a random weight. The weights are choosen randomly between -1 and +1.
   *
   * @param ns `List[Neuron]` List of input Neurons
   * @return `List[Dendrite]` which is added to the immutable `dendrites` property on instantiation of this Neuron
   */
  private def connect(ns: List[Neuron]): List[Dendrite] = ns.map { n =>
    new Dendrite(n, (new Random).nextDouble * 2 * pow(ns.size, -0.5) - 1)
  }

  override def toString = s"${dendrites.mkString(",")} $name\n"

}

object Neuron {

  def apply(name: String, lower: List[Neuron], activation: ActivationFunctions.Value): Neuron = activation match {
    case HyperbolicTangent => new Neuron(name, lower) with HyperbolicTangent
    case Sigmoid => new Neuron(name, lower) with Sigmoid
  }

}