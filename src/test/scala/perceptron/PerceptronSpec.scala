package perceptron


import org.scalatest._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import scala.util.Random
import org.scalatest.matchers.ShouldMatchers
import ActivationFunctions._

@RunWith(classOf[JUnitRunner])
class PerceptronSpec extends FlatSpec with ShouldMatchers {

  "A Perceptron" should "be able to solve XOR" in {

    // The index of the list represents the layer # and the
    // integer value the amount of neurons in that layer.
    val net = new Perceptron(List(2,3,2,1))

    // training
    for (i <- 1 to 150) {
      net.train(List(1.0, 1.0), List(-1.0))
      net.train(List(-1.0, -1.0), List(-1.0))
      net.train(List(1.0, -1.0), List(1.0))
      net.train(List(-1.0, 1.0), List(1.0))
    }

    // run
    net.run(List(1.0, 1.0)).head should be (-1.0 plusOrMinus 0.2)
    net.run(List(-1.0, 1.0)).head should be (1.0 plusOrMinus 0.2)
    net.run(List(1.0, -1.0)).head should be (1.0 plusOrMinus 0.2)
    net.run(List(-1.0, -1.0)).head should be (-1.0 plusOrMinus 0.2)
  }

  it should "be able to solve OR" in {

    val net = new Perceptron(List(2,3,2,1))

    // training
    for (i <- 1 to 150) {
      net.train(List(1.0, 1.0), List(1.0))
      net.train(List(1.0, -1.0), List(1.0))
      net.train(List(-1.0, 1.0), List(1.0))
      net.train(List(-1.0, -1.0), List(-1.0))
    }

    // run
    net.run(List(1.0, 1.0)).head should be (1.0 plusOrMinus 0.2)
    net.run(List(-1.0, 1.0)).head should be (1.0 plusOrMinus 0.2)
    net.run(List(1.0, -1.0)).head should be (1.0 plusOrMinus 0.2)
    net.run(List(-1.0, -1.0)).head should be (-1.0 plusOrMinus 0.2)
  }


  it should "be able to solve AND" in {

    val net = new Perceptron(List(2,3,2,1))

    // training
    for (i <- 1 to 150) {
      net.train(List(1.0, 1.0), List(1.0))
      net.train(List(1.0, -1.0), List(-1.0))
      net.train(List(-1.0, 1.0), List(-1.0))
      net.train(List(-1.0, -1.0), List(-1.0))
    }

    // run
    net.run(List(1.0, 1.0)).head should be (1.0 plusOrMinus 0.2)
    net.run(List(-1.0, 1.0)).head should be (-1.0 plusOrMinus 0.2)
    net.run(List(1.0, -1.0)).head should be (-1.0 plusOrMinus 0.2)
    net.run(List(-1.0, -1.0)).head should be (-1.0 plusOrMinus 0.2)
  }

  it should "be able to solve NOT" in {

    val net = new Perceptron(List(2,3,2,1))

    // training
    for (i <- 1 to 150) {
      net.train(List(1.0, 1.0), List(-1.0))
      net.train(List(1.0, -1.0), List(-1.0))
      net.train(List(-1.0, 1.0), List(-1.0))
      net.train(List(-1.0, -1.0), List(1.0))
    }

    // run
    net.run(List(1.0, 1.0)).head should be (-1.0 plusOrMinus 0.2)
    net.run(List(-1.0, 1.0)).head should be (-1.0 plusOrMinus 0.2)
    net.run(List(1.0, -1.0)).head should be (-1.0 plusOrMinus 0.2)
    net.run(List(-1.0, -1.0)).head should be (1.0 plusOrMinus 0.2)
  }

  it should "be able to solve NOT with Sigmoid activation function" in {

    val net = new Perceptron(List(2,2,1), Sigmoid)

    // training
    for (i <- 1 to 1000) {
      net.train(List(1., 1.), List(0.))
      net.train(List(1., 0.), List(0.))
      net.train(List(0., 1.), List(0.))
      net.train(List(0., 0.), List(1.))
    }

    // run
    net.run(List(1., 1.)).head should be (0. plusOrMinus 0.2)
    net.run(List(0., 1.)).head should be (0. plusOrMinus 0.2)
    net.run(List(1., 0.)).head should be (0. plusOrMinus 0.2)
    net.run(List(0., 0.)).head should be (1. plusOrMinus 0.2)
  }
}