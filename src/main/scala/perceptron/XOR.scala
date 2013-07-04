package perceptron

import scala.util.Random

object XOR extends App {
  val net = new Perceptron(List(2,3,2,1), new Random)
  println(net)
  for (i <- 1 to 150) {
    net.train(List(1, 1), List(-1))
    net.train(List(-1, -1), List(-1))
    net.train(List(1, -1), List(1))
    net.train(List(-1, 1), List(1))
    if (i % 33 == 0) println(net)
  }
  println("Training done.")
  println("** Output for (1,1) " + net.output(List(1, 1)))
  println("** Output for (1,-1) " + net.output(List(1, -1)))
  println("** Output for (-1,1) " + net.output(List(-1, 1)))
  println("** Output for (-1,-1) " + net.output(List(-1, -1)))
}
