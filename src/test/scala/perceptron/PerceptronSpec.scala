package perceptron


import org.scalatest._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.matchers.ShouldMatchers
import ActivationFunctions._
import scala.concurrent.{Future, ExecutionContext}
import ExecutionContext.Implicits.global


@RunWith(classOf[JUnitRunner])
class PerceptronSpec extends FlatSpec with ShouldMatchers {

  "A Perceptron" should "be able to solve XOR using hyperbolic tangent activation" in {

    // The index of the list represents the layer # and the
    // integer value the amount of neurons in that layer.
    val net = new Perceptron(List(2,3,2,1), HyperbolicTangent)

    // training
    net.train(
      List(
        Pattern(List(1.0, 1.0), List(-1.0)),
        Pattern(List(-1.0, -1.0), List(-1.0)),
        Pattern(List(1.0, -1.0), List(1.0)),
        Pattern(List(-1.0, 1.0), List(1.0))
      ),
      iterations = 150
    )

    // run
    Util.await(net.run(List(1.0, 1.0))).head should be (-1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, 1.0))).head should be (1.0 plusOrMinus 0.2)
    Util.await(net.run(List(1.0, -1.0))).head should be (1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, -1.0))).head should be (-1.0 plusOrMinus 0.2)
  }

  it should "be able to solve OR using hyperbolic tangent activation" in {

    val net = new Perceptron(List(2,3,2,1), HyperbolicTangent)

    // training
    net.train(
      List(
        Pattern(List(1.0, 1.0), List(1.0)),
        Pattern(List(1.0, -1.0), List(1.0)),
        Pattern(List(-1.0, 1.0), List(1.0)),
        Pattern(List(-1.0, -1.0), List(-1.0))
      ),
      iterations = 150
    )

    // run
    Util.await(net.run(List(1.0, 1.0))).head should be (1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, 1.0))).head should be (1.0 plusOrMinus 0.2)
    Util.await(net.run(List(1.0, -1.0))).head should be (1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, -1.0))).head should be (-1.0 plusOrMinus 0.2)
  }


  it should "be able to solve AND using hyperbolic tangent activation" in {

    val net = new Perceptron(List(2,3,2,1), HyperbolicTangent)

    // training
    net.train(
      List(
        Pattern(List(1.0, 1.0), List(1.0)),
        Pattern(List(1.0, -1.0), List(-1.0)),
        Pattern(List(-1.0, 1.0), List(-1.0)),
        Pattern(List(-1.0, -1.0), List(-1.0))
      ),
      iterations = 150
    )

    // run
    Util.await(net.run(List(1.0, 1.0))).head should be (1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, 1.0))).head should be (-1.0 plusOrMinus 0.2)
    Util.await(net.run(List(1.0, -1.0))).head should be (-1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, -1.0))).head should be (-1.0 plusOrMinus 0.2)
  }

  it should "be able to solve NOT using hyperbolic tangent activation" in {

    val net = new Perceptron(List(2,3,2,1), HyperbolicTangent)

    // training
    net.train(
      List(
        Pattern(List(1.0, 1.0), List(-1.0)),
        Pattern(List(1.0, -1.0), List(-1.0)),
        Pattern(List(-1.0, 1.0), List(-1.0)),
        Pattern(List(-1.0, -1.0), List(1.0))
      ),
      iterations = 150
    )

    // run
    Util.await(net.run(List(1.0, 1.0))).head should be (-1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, 1.0))).head should be (-1.0 plusOrMinus 0.2)
    Util.await(net.run(List(1.0, -1.0))).head should be (-1.0 plusOrMinus 0.2)
    Util.await(net.run(List(-1.0, -1.0))).head should be (1.0 plusOrMinus 0.2)
  }

  it should "be able to solve AND with sigmoid activation" in {

    val net = new Perceptron(List(2,2,1))

    // training
    net.train(
      List(
        Pattern(List(1., 1.), List(1.)),
        Pattern(List(1., 0.), List(0.)),
        Pattern(List(0., 1.), List(0.)),
        Pattern(List(0., 0.), List(0.))
      ),
      iterations = 1000
    )

    // run
    Util.await(net.run(List(1., 1.))).head should be (1. plusOrMinus 0.2)
    Util.await(net.run(List(0., 1.))).head should be (0. plusOrMinus 0.2)
    Util.await(net.run(List(1., 0.))).head should be (0. plusOrMinus 0.2)
    Util.await(net.run(List(0., 0.))).head should be (0. plusOrMinus 0.2)
  }

  it should "be able to solve OR with sigmoid activation" in {

    val net = new Perceptron(List(2,2,1))

    // training
    net.train(
      List(
        Pattern(List(1., 1.), List(1.)),
        Pattern(List(1., 0.), List(1.)),
        Pattern(List(0., 1.), List(1.)),
        Pattern(List(0., 0.), List(0.))
      ),
      iterations = 1000
    )

    // run
    Util.await(net.run(List(1., 1.))).head should be (1. plusOrMinus 0.2)
    Util.await(net.run(List(0., 1.))).head should be (1. plusOrMinus 0.2)
    Util.await(net.run(List(1., 0.))).head should be (1. plusOrMinus 0.2)
    Util.await(net.run(List(0., 0.))).head should be (0. plusOrMinus 0.2)
  }

  it should "be able to solve NOT with sigmoid activation" in {

    val net = new Perceptron(List(2,2,1))

    // training
    net.train(
      List(
        Pattern(List(1., 1.), List(0.)),
        Pattern(List(1., 0.), List(0.)),
        Pattern(List(0., 1.), List(0.)),
        Pattern(List(0., 0.), List(1.))
      ),
      iterations = 1000
    )

    // run
    Util.await(net.run(List(1., 1.))).head should be (0. plusOrMinus 0.2)
    Util.await(net.run(List(0., 1.))).head should be (0. plusOrMinus 0.2)
    Util.await(net.run(List(1., 0.))).head should be (0. plusOrMinus 0.2)
    Util.await(net.run(List(0., 0.))).head should be (1. plusOrMinus 0.2)
  }

  it should "be able to solve XOR with sigmoid activation" in {

    val net = new Perceptron(List(2,3,1))

    // training
    net.train(
      List(
        Pattern(List(1., 1.), List(0.)),
        Pattern(List(1., 0.), List(1.)),
        Pattern(List(0., 1.), List(1.)),
        Pattern(List(0., 0.), List(0.))
      ),
      iterations = 5000
    )

    // run
    Util.await(net.run(List(1., 1.))).head should be (0. plusOrMinus 0.2)
    Util.await(net.run(List(0., 1.))).head should be (1. plusOrMinus 0.2)
    Util.await(net.run(List(1., 0.))).head should be (1. plusOrMinus 0.2)
    Util.await(net.run(List(0., 0.))).head should be (0. plusOrMinus 0.2)
  }

  it should "be able to recognize ten patterns using sigmoid activation" in {

    val net = new Perceptron(List(50, 100, 10), momentum = 0.1)

    def proofActiveNeuron(index: Int, input: List[Double], net: Perceptron) =
      net.run(input) map { result =>
        for ((value, i) <- result zipWithIndex)
          if (i == index) value should be (1. plusOrMinus .2) else value should be (0. plusOrMinus .2)
      }

    def activeNeuron(output: List[Double]) = output indexOf(1.)

    val patterns = List(
      Pattern(List(.0,1.,1.,1.,.0,1.,1.,.0,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,.0,.0,1.,.0,1.,1.,1.,.0), List(1.,.0,.0,.0,.0,.0,.0,.0,.0,.0)),
      Pattern(List(1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,1.,1.,1.,1.), List(.0,1.,.0,.0,.0,.0,.0,.0,.0,.0)),
      Pattern(List(1.,1.,1.,.0,.0,.0,.0,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,1.,1.,1.,1.,.0,1.,.0,1.,1.,.0,1.,1.,1.,1.,1.), List(.0,.0,1.,.0,.0,.0,.0,.0,.0,.0)),
      Pattern(List(1.,1.,1.,1.,.0,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,1.,.0,.0,.0,.0,.0,.0)),
      Pattern(List(1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,1.,1.,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0), List(.0,.0,.0,.0,1.,.0,.0,.0,.0,.0)),
      Pattern(List(1.,1.,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,.0,.0,1.,.0,.0,.0,.0)),
      Pattern(List(.0,1.,1.,1.,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,1.,1.,.0,1.,1.,1.,1.,1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,.0,1.,1.,1.,1.), List(.0,.0,.0,.0,.0,.0,1.,.0,.0,.0)),
      Pattern(List(1.,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0), List(.0,.0,.0,.0,.0,.0,.0,1.,.0,.0)),
      Pattern(List(1.,1.,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,1.,1.,1.,.0,1.,1.,1.,1.,1.,1.,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,.0,.0,.0,.0,.0,1.,.0)),
      Pattern(List(1.,1.,1.,1.,1.,1.,1.,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.), List(.0,.0,.0,.0,.0,.0,.0,.0,.0,1.))
    )

    // training
    net.train(patterns, iterations = 150)

    // run
    patterns map { case Pattern(input, output) => proofActiveNeuron(activeNeuron(output), input, net) }

  }
}