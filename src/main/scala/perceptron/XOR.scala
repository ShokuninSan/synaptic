package perceptron

import scala.util.Random

object XOR extends App {
  // The index of the list represents the layer # and the
  // integer value the amount of neurons in that layer.
  val net = new Perceptron(List(2,3,2,1))
  println("The perceptron:")
  println(net)
  for (i <- 1 to 150) {
    net.train(List(1.0, 1.0), List(-1.0))
    net.train(List(-1.0, -1.0), List(-1.0))
    net.train(List(1.0, -1.0), List(1.0))
    net.train(List(-1.0, 1.0), List(1.0))
    //if (i % 33 == 0) println(net)
  }
  println("Training done.")
  println("** Output for (1,1) " + net.run(List(1.0, 1.0)))
  println("** Output for (1,-1) " + net.run(List(1.0, -1.0)))
  println("** Output for (-1,1) " + net.run(List(-1.0, 1.0)))
  println("** Output for (-1,-1) " + net.run(List(-1.0, -1.0)))
}