package perceptron


class Dendrite(neuron: Neuron, private var weight: Double) {

  def input: Double =  neuron.out * weight

  def updateError(delta: Double) = neuron updateError(delta * weight)

  def adjust(adjustment: Double) = weight += adjustment * neuron.out

  override def toString = "--[" + weight + "]-->" + neuron

}
