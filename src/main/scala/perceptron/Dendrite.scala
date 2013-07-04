package perceptron

class Dendrite(n: Neuron, w: Double) {
  // the input neuron
  val neuron = n
  // weight of the signal
  var weight = w
 
  def input = weight * neuron.out;
 
  def updateError(delta: Double) {
    neuron.updateError(delta * weight)
  }
 
  def adjust(adjustment: Double) {
    weight += adjustment * neuron.out
  }
 
  override def toString = "--["+weight+"]-->"+neuron.name
}
