package perceptron

import scala.concurrent.{Future,ExecutionContext}
import ExecutionContext.Implicits.global
import scala.math._

case class Pattern(input: List[Double], output: List[Double])

trait BackpropagationTrainer {

  this: Perceptron =>

  /**
   * The training function
   *
   * Training of the perceptron is done (simplified) in two steps:
   *
   *  1. Run with the given input:
   *    Sets the output of the input-neurons to the given values and calls the
   *    'fire' function of each neuron within each layer of the network. The
   *    'fire' function of the Neurons are the activation-functions (i.e.
   *    tangens hyperbolicus or logistic/sigmoid function). The run function below
   *    returns the output of the the last neuron, the output neuron.
   *
   *  2. Backpropagation - step 1:
   *    Take the last neuron (the output neuron) and updates the 'error' fields
   *    on each neuron of the network recursively by adding a delta, which is
   *    calculated on the Axon (for hidden layers) by multiplication of it's
   *    weight respectively on the output Neuron by subtraction of the expected
   *    output value minus the actual output value. This is done by execution of
   *    the [[perceptron.Neuron.updateError()]] functions.
   *
   *  3. Backpropagation - step 2:
   *    Calculates the bias via the derivative activation function multiplied by
   *    the corresponding result of the 'error' calculation of step 1.
   *
   * @param patterns The training data
   * @param iterations Number of max iterations
   */
  def train(patterns: List[Pattern], iterations: Int): Unit = {
    def go(inputs: List[Double], outputs: List[Double]): Future[List[Double]] =
      for {
        output <- run(inputs)
        _ <- backPropagate(outputs)
        _ <- applyDeltaRule
      } yield output
    def satisfactory(input: List[Double], output: List[Double]): Boolean = {
      println(s"expected($output), actual($input)")
      ((input zip output).foldLeft(0.0)((a,b) => a + abs(b._1 + b._2)) <= 0.01)
    }
    for {
      i <- (1 to iterations).reverse
      t <-
        patterns map {
          _ match {
            case Pattern(input, output) => (Util.await(go(input, output)), output)
          }
        }
      _ <- if (autoAdjust) adjustLearningRate(i, iterations) else Nil
      if (!(satisfactory _).tupled(t))
    } yield ()
  }

  private def backPropagate(outs: List[Double]): Future[List[Double]] =
    Future.sequence(layers.last.zip(0 until outs.length) map (t => t._1.backPropagate(outs(t._2))))

  private def applyDeltaRule: Future[List[Double]] =
    Future.sequence(layers flatMap { _ map (_ applyDeltaRule) })

  private def adjustLearningRate(iteration: Int, iterations: Int) = {
    val i = BigDecimal(iteration).setScale(2, BigDecimal.RoundingMode.HALF_UP)
    val is = BigDecimal(iterations).setScale(2, BigDecimal.RoundingMode.HALF_UP)
    layers map { _ map
      { neuron =>
        val initialEta = BigDecimal(neuron.initialLearningRate).setScale(2, BigDecimal.RoundingMode.HALF_UP)
        val newEta = ((initialEta/100) * ((i/is * 100))).toDouble
        if (newEta >= 0.1)
          neuron.learningRate = newEta
        else
          neuron.learningRate = 0.1
      }
    }
  }

}