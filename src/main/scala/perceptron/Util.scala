package perceptron

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._

object Util {

  def await(future: Future[List[Double]]): List[Double] = Await.result(future, Duration.Inf)

}
