package perceptron

import scala.concurrent.{Future, Await}
import scala.concurrent.duration._

object XOR extends App {

  val net = new Perceptron(List(2,3,2,1))

  def await(future: Future[List[Double]]): List[Double] = Await.result(future, 5 seconds)

  for (i <- 1 to 150) {
    await(net.train(List(1.0, 1.0), List(-1.0)))
    await(net.train(List(-1.0, -1.0), List(-1.0)))
    await(net.train(List(1.0, -1.0), List(1.0)))
    await(net.train(List(-1.0, 1.0), List(1.0)))
  }

  println("Training done.")
  println("** Output for (1,1) "   + await(net.run(List(1.0, 1.0))))
  println("** Output for (1,-1) "  + await(net.run(List(1.0, -1.0))))
  println("** Output for (-1,1) "  + await(net.run(List(-1.0, 1.0))))
  println("** Output for (-1,-1) " + await(net.run(List(-1.0, -1.0))))
}