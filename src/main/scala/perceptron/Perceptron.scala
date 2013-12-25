package perceptron

import ActivationFunctions._
import scala.concurrent.{Await, Future, ExecutionContext}
import ExecutionContext.Implicits.global
import scala.concurrent.duration._

class Perceptron(layout: List[Int], activation: ActivationFunctions.Value = HyperbolicTangent) {

  val layers = build(layout)

  def run(ins: List[Double]): Future[List[Double]] = {
    // Set output of input-neurons to the given values
    layers.head.zip(ins).foreach { case (n, in) => n feed in }
    // Call 'output' function of each neuron on each but the first layer
    // Note that we're just interested in the output neurons
    val futures = (layers.tail map { _ map { _ fire } }).last
    Future.sequence(futures)
  }

  /**
   * The training function
   *
   * Training of the perceptron is done (simplified) in two steps:
   *
   *  1. Run with the given input:
   *    Sets the output of the input-neurons to the given values and calls the
   *    'fire' function of each neuron within each layer of the network. The
   *    'fire' function of the Neurons are the activation-functions (i.e.
   *    tangens hyperbolicus or logistic/sigmoid function). The run function below
   *    returns the output of the the last neuron, the output neuron.
   *
   *  2. Backpropagation - step 1:
   *    Take the last neuron (the output neuron) and updates the 'error' fields
   *    on each neuron of the network recursively by adding a delta, which is
   *    calculated on the Axon (for hidden layers) by multiplication of it's
   *    weight respectively on the output Neuron by subtraction of the expected
   *    output value minus the actual output value. This is done by execution of
   *    the [[perceptron.Neuron.updateError()]] functions.
   *
   *  3. Backpropagation - step 2:
   *    Calculates the bias via the derivative activation function multiplied by
   *    the corresponding result of the 'error' calculation of step 1.
   *
   * @param ins Sets the output of the input-neurons to the given values
   * @param outs Expected output values
   */
  def train(ins: List[Double], outs: List[Double]): Future[List[Double]] =
    for {
      _ <- run(ins)
      _ <- backPropagate(outs)
      delta <- adjust
    } yield delta

  private def backPropagate(outs: List[Double]): Future[List[Double]] = Future.sequence(layers.last.zip(0 until outs.length) map (t => t._1.backPropagate(outs(t._2))))

  private def adjust: Future[List[Double]] = Future.sequence(layers flatMap { _ map (_ adjust) })

  private def build(layout: List[Int]): List[List[Neuron]] = layout.zip(1 to layout.size).foldLeft(List(List[Neuron]())) {
      case (previousLayer, (neuronIndex, layer)) => buildLayer(s"L$layer", neuronIndex, previousLayer.head) :: previousLayer
    }.reverse.tail

  private def buildLayer(name: String, n: Int, lower: List[Neuron]) = (0 until n) map { n => Neuron(s"$name-N$n", lower, activation)} toList

  override def toString = layers.mkString("\n")
}
