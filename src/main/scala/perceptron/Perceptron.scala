package perceptron

import ActivationFunctions._
import scala.concurrent.{Future, ExecutionContext}
import ExecutionContext.Implicits.global

class Perceptron(layout: List[Int], activation: ActivationFunctions.Value = HyperbolicTangent, val learningRate: Double = 0.1, val autoAdjust: Boolean = false, val momentum: Double = .9) extends BackpropagationTrainer {

  val layers: List[List[Neuron]] = spanNetwork(layout)

  def run(ins: List[Double]): Future[List[Double]] = {
    // Set output of input-neurons to the given values
    layers.head.zip(ins).foreach { case (n, in) => n feed in }
    // Call 'output' function of each neuron on each but the first layer
    // Note that we're just interested in the output neurons
    val futures = (layers.tail map { _ map { _ fire } }).last
    Future.sequence(futures)
  }

  private def spanNetwork(layout: List[Int]): List[List[Neuron]] =
    layout.zip(1 to layout.size).foldLeft(List(List[Neuron]())) {
      case (previousLayer, (neuronIndex, layer)) =>
        createLayer(s"Layer($layer)", neuronIndex, previousLayer.head) :: previousLayer
    }.reverse.tail

  private def createLayer(name: String, n: Int, lower: List[Neuron]): List[Neuron] =
    (0 until n) map { n => Neuron(s"$name-Neuron($n)", lower, activation, learningRate, momentum)} toList

  override def toString = layers.mkString("\n")
}
