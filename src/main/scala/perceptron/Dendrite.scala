package perceptron
import scala.concurrent.{ExecutionContext, Future}

class Dendrite(neuron: Neuron, private var weight: Double) {

  /**
   * Computes the input of the upstream Neuron.
   *
   * This function computes the input of the connected upstream Neuron. The computation is based upon the formula
   *
   *   [[perceptron.Neuron.out]] * this.weight
   *
   * The `netj` (the overall netto input) of the Dentrites' Neuron is computed by [[perceptron.Neuron.input]] which is
   * implemented with a simple
   *
   *   {{{ dendrites.map(_.input).sum + bias }}}
   *
   * @return the computed netto input of the upstream Neuron.
   */
  def input: Double =  neuron.out * weight

  /**
   * This function calls the `updateError` function of the upstream Neuron.
   *
   * This function is indented to be called from within the [[perceptron.Neuron.updateError()]] function, where the
   * Neron computes the deltas for each of it's upstream Neurons by executions of the following statement:
   *
   *   {{{ dendrites.foreach(_.updateError(delta)) }}}
   *
   * It is part of the backpropagation mechansim, where as a first step, all deltas are adjusted within the hidden
   * layers.
   *
   * @param delta The computed delta (deltak) of the downstream layer (the layer with index k)
   * @return the delta/error of the upstream Neuron
   */
  def updateError(delta: Double): Future[Double] = neuron updateError(delta * weight)

  /**
   * Adjusts the weight computed by the delta rule.
   *
   * This function adjusts the weight according to the backpropagation rule which is analogous to the Delta Rule:
   *
   *   delta wij = eta * oi * deltaj
   *
   * Where `deltaj` is the result of
   *
   *   f'(netj) (tj-oj)           ... in case of an output Neuron, or
   *   f'(netj) sum(deltak * wjk) ... in case of a hidden Neuron
   *
   * This function is intended to be called from within the [[perceptron.Neuron.applyDeltaRule]] function.
   *
   * @param adjustment The result of the partial computation of the backpropagation rule which has all except the output
   *                   of the upstream Neuron, thus it's the result of i.e. `deltawij = eta * deltaj * _` where the _ is
   *                   the output which gets applied whithin this function finally.
   */
  def applyDeltaRule(adjustment: Double): Unit = weight += adjustment * neuron.out

  override def toString = s"$neuron - [ $weight ] ->"

}
