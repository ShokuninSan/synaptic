package perceptron

class Dendrite(neuron: Neuron, private var weight: Double) {

  def input = weight * neuron.out

  def updateError(delta: Double) = neuron.updateError(delta * weight)

  def adjust(adjustment: Double) = weight += adjustment * neuron.out

  override def toString = "--[" + weight + "]-->" + neuron.name

}
