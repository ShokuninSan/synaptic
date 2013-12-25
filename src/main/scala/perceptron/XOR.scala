package perceptron

import scala.concurrent.{Future, Await}
import scala.concurrent.duration._

object XOR extends App {

  val net = new Perceptron(List(2,3,2,1))

  def await(future: Future[List[Double]]): List[Double] = Await.result(future, 5 seconds)

  net.train(
    List(
      Pattern(List(1., 1.), List(0.)),
      Pattern(List(1., 0.), List(1.)),
      Pattern(List(0., 1.), List(1.)),
      Pattern(List(0., 0.), List(0.))
    ),
    iterations = 150
  )

  println("Training done.")
  println("** Output for (1,1) "   + await(net.run(List(1.0, 1.0))))
  println("** Output for (1,-1) "  + await(net.run(List(1.0, -1.0))))
  println("** Output for (-1,1) "  + await(net.run(List(-1.0, 1.0))))
  println("** Output for (-1,-1) " + await(net.run(List(-1.0, -1.0))))
}