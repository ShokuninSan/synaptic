package perceptron


import org.scalatest._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.matchers.ShouldMatchers
import ActivationFunctions._
import scala.concurrent.{Future, Await}


@RunWith(classOf[JUnitRunner])
class PerceptronSpec extends FlatSpec with ShouldMatchers {

  "A Perceptron" should "be able to solve XOR" in {

    // The index of the list represents the layer # and the
    // integer value the amount of neurons in that layer.
    val net = new Perceptron(List(2,3,2,1))

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

  it should "be able to solve OR" in {

    val net = new Perceptron(List(2,3,2,1))

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


  it should "be able to solve AND" in {

    val net = new Perceptron(List(2,3,2,1))

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

  it should "be able to solve NOT" in {

    val net = new Perceptron(List(2,3,2,1))

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

  it should "be able to solve NOT with Sigmoid activation function" in {

    val net = new Perceptron(List(2,2,1), Sigmoid)

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

  it should "be able to solve XOR with Sigmoid activation function" in {

    val net = new Perceptron(List(2,3,1), Sigmoid)

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

  it should "be able to recognize five letters" in {
    val net = new Perceptron(List(50, 70, 5), Sigmoid)

    // training
    net.train(
      List(
        Pattern(List(.0,1.,1.,1.,.0,1.,1.,.0,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,.0,.0,1.,.0,1.,1.,1.,.0), List(1.,.0,.0,.0,.0)),
        Pattern(List(1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,1.,1.,1.,1.), List(.0,1.,.0,.0,.0)),
        Pattern(List(1.,1.,1.,.0,.0,.0,.0,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,1.,1.,1.,1.,.0,1.,.0,1.,1.,.0,1.,1.,1.,1.,1.), List(.0,.0,1.,.0,.0)),
        Pattern(List(1.,1.,1.,1.,.0,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,1.,.0)),
        Pattern(List(1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,1.,1.,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0), List(.0,.0,.0,.0,1.))
      ),
      iterations = 1000
    )

    // run
    Util.await(net.run(List(1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,1.,1.,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0))) should be (List(.0,.0,.0,.0,1.,.0,.0,.0,.0,.0))
  }

//  it should "be able to recognize ten letters" in {
//    val net = new Perceptron(List(50, 70, 50, 10), Sigmoid)
//
//    // training
//    net.train(
//      List(
//        Pattern(List(.0,1.,1.,1.,.0,1.,1.,.0,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,.0,.0,1.,.0,1.,1.,1.,.0), List(1.,.0,.0,.0,.0,.0,.0,.0,.0,.0)),
//        Pattern(List(1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,1.,1.,1.,1.), List(.0,1.,.0,.0,.0,.0,.0,.0,.0,.0)),
//        Pattern(List(1.,1.,1.,.0,.0,.0,.0,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,1.,1.,1.,1.,.0,1.,.0,1.,1.,.0,1.,1.,1.,1.,1.), List(.0,.0,1.,.0,.0,.0,.0,.0,.0,.0)),
//        Pattern(List(1.,1.,1.,1.,.0,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,.0,.0,1.,1.,.0,.0,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,1.,.0,.0,.0,.0,.0,.0)),
//        Pattern(List(1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,1.,1.,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0), List(.0,.0,.0,.0,1.,.0,.0,.0,.0,.0)),
//        Pattern(List(1.,1.,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,.0,.0,1.,.0,.0,.0,.0)),
//        Pattern(List(.0,1.,1.,1.,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,1.,1.,.0,1.,1.,1.,1.,1.,1.,1.,.0,.0,1.,1.,1.,.0,.0,1.,.0,1.,1.,1.,1.), List(.0,.0,.0,.0,.0,.0,1.,.0,.0,.0)),
//        Pattern(List(1.,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,.0,.0,.0), List(.0,.0,.0,.0,.0,.0,.0,1.,.0,.0)),
//        Pattern(List(1.,1.,1.,1.,1.,1.,.0,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,1.,1.,1.,.0,1.,1.,1.,1.,1.,1.,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,1.,1.,1.), List(.0,.0,.0,.0,.0,.0,.0,.0,1.,.0)),
//        Pattern(List(1.,1.,1.,1.,1.,1.,1.,.0,.0,1.,1.,.0,.0,.0,1.,1.,1.,1.,1.,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.), List(.0,.0,.0,.0,.0,.0,.0,.0,.0,1.))
//      ),
//      iterations = 1000
//    )
//
//    // run
//    Util.await(net.run(List(1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,.0,.0,1.,.0,1.,1.,1.,1.,1.,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0,.0,.0,.0,1.,.0))) should be (List(.0,.0,.0,.0,1.,.0,.0,.0,.0,.0))
//  }
}