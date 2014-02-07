package wasabi.perceptron

import scala.concurrent.{Future,ExecutionContext}
import ExecutionContext.Implicits.global
import scala.math._

case class Pattern(input: List[Double], output: List[Double])

trait BackpropagationTrainer {

  this: Perceptron =>

  def train(patterns: List[Pattern], iterations: Int): Unit =
    for {
      i <- (1 to iterations).reverse
      t <- epoch(patterns)
    } yield
      if(autoAdjust) adjustLearningRate(i, iterations)

  private def epoch(patterns: List[Pattern]) =
    patterns map {
      case Pattern(input, output) => (Util.await(go(input, output)), output)
    }

  private def go(input: List[Double], output: List[Double]): Future[List[Double]] =
    for {
      o <- run(input)
      _ <- backPropagate(output)
    } yield o

  private def backPropagate(outs: List[Double]) =
    for {
      _ <- updateError(outs)
      _ <- applyDeltaRule
    } yield ()

  private def updateError(outs: List[Double]): Future[List[Double]] =
    Future.sequence(layers.last.zip(0 until outs.length) map {
      case (neuron, i) => neuron.updateError(outs(i) - neuron.output)
    })

  private def applyDeltaRule: Future[List[Double]] =
    Future.sequence(layers flatMap { _ map (_ applyDeltaRule) })

  private def adjustLearningRate(iteration: Int, iterations: Int) = {
    val i = BigDecimal(iteration).setScale(2, BigDecimal.RoundingMode.HALF_UP)
    val is = BigDecimal(iterations).setScale(2, BigDecimal.RoundingMode.HALF_UP)
    layers foreach { _ map
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