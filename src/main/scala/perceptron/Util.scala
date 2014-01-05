package perceptron

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.util.Random
import scala.math._

object Util {

  def await(future: Future[List[Double]]): List[Double] = Await.result(future, Duration.Inf)

  def randomBiasWeight = (new Random).nextDouble * 2.0 - 1.0

  def randomNeuronWeight(inputNeuronCount: Int) = (new Random).nextDouble * 2 * pow(inputNeuronCount, -0.5) - 1

}
