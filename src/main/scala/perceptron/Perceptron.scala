package perceptron

import scala.util.Random
import ActivationFunctions._

class Perceptron(layout: List[Int], activation: ActivationFunctions.Value = HyperbolicTangent) {
  val layers = build(layout)

  def run(ins: List[Double]) = {
    // Set output of input-neurons to the given values
    layers.head.zip(ins).foreach { case (n, in) => n.out = in }
    // Call 'output' function of each neuron on each layer
    layers.tail.foldLeft(Nil: List[Double]) { (_, l) => l.map(_.output) }
  }

  def train(ins: List[Double], outs: List[Double]) = {
    // 1. Run with the given input:
    // Sets the output of the input-neurons to the given values and calls the
    // 'output' function of each neuron within each layer of the network. The
    // 'output' function of the Neurons are the activation-functions (i.e.
    // tangens hyperbolicus or logistic function).
    // The run function below returns the output of the the last neuron,
    // the output neuron.
    val output = run(ins)
    // 2. Backpropagation:
    // Take the last neuron (the output neuron) and update the 'error' fields
    // on each neuron of the network recursively by adding a delta, which is
    // calculated on the Axon (for hidden layers) by multiplication of it's weight
    // respectively on the output Neuron by subtraction of the expected output
    // value minus the actual output value.
    layers.last.zip(0 until outs.length).foreach {case (n, m) => n.backpropagate(outs(m))}
    layers flatMap {
      _ map (_ adjust)
    }
  }

  override def toString = layers.mkString("\n")

  private def build(layout: List[Int]) =
    layout.zip(1 to layout.size).foldLeft(List(List[Neuron]())) {
      case (previousLayer, (neuronIndex, layer)) => buildLayer("L"+layer, neuronIndex, previousLayer.head) :: previousLayer
    }.reverse.tail

  private def buildLayer(name: String, n: Int, lower: List[Neuron]) =
    (0 until n) map { n =>
      activation match {
        case HyperbolicTangent => new NeuronImpl(name+"N"+n, lower) with HyperbolicTangent
        case Sigmoid => new NeuronImpl(name+"N"+n, lower) with Sigmoid
      }
    } toList
}
