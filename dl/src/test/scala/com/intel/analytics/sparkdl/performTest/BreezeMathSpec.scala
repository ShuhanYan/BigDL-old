package com.intel.analytics.sparkdl.performTest
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import org.scalatest.FlatSpec

/**
  * Created by yansh on 16-9-27.
  */
class BreezeMathSpec  extends FlatSpec{
  def init(length:Int,rangeMin:Float,rangeMax:Float,interval:Float): DenseMatrix[Float] ={
    val result = new DenseMatrix[Float](length, length)
    var const = rangeMin
    for (i <- 0 until(length)){
      for (j <- 0 until(length)) {
        if (const > rangeMax) const = rangeMin
        result.update(i, j, const)
        const = const+interval
      }
    }
    result
  }

  val Seed = 100
  val sizeLarge = 4096
  val matrixLargeLeft = init(sizeLarge,-1000,1000,0.5f)
  val matrixLargeRight = init(sizeLarge,-1000,1000,0.5f)
  val vectorLarge = DenseVector.rand(sizeLarge ,Rand.uniform.map(_.toFloat))
  val sizeMid = 512
  val matrixMidLeft = init(sizeMid,-1000,1000,0.5f)
  val matrixMidRight = init(sizeMid,-1000,1000,0.5f)
  val vectorMid = DenseVector.rand(sizeMid ,Rand.uniform.map(_.toFloat))
  val sizeSmall = 32
  val matrixSmallLeft = init(sizeSmall,-500,500,1)
  val matrixSmallRight = init(sizeSmall,-500,500,1)
  val vectorSmall = DenseVector.rand(sizeSmall ,Rand.uniform.map(_.toFloat))
  val scalar = 5

  var testCase = " Breeze 4096 * 4096 matrix add operation"
  TestUtils.testMathOperation(() => matrixLargeLeft+matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix add operation"
  TestUtils.testMathOperation(() => matrixMidLeft+matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix add operation"
  TestUtils.testMathOperation(() => matrixSmallLeft+matrixSmallRight, testCase)

  testCase = " Breeze 4096 * 4096 matrix minus operation"
  TestUtils.testMathOperation(() => matrixLargeLeft-matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix minus operation"
  TestUtils.testMathOperation(() => matrixMidLeft-matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix minus operation"
  TestUtils.testMathOperation(() => matrixSmallLeft-matrixSmallRight, testCase)

  testCase = " Breeze 4096 * 4096 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixLargeLeft*matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixMidLeft*matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixSmallLeft * matrixSmallRight, testCase)

  testCase = " Breeze 4096 * 4096 matrix divide operation"
  TestUtils.testMathOperation(() => matrixLargeLeft/matrixLargeRight, testCase, 1)

  testCase = " Breeze 512 * 512 matrix divide operation"
  TestUtils.testMathOperation(() => matrixMidLeft/matrixMidRight, testCase)

  testCase = " Breeze 32 * 32 matrix divide operation"
  TestUtils.testMathOperation(() => matrixSmallLeft/matrixSmallRight, testCase)

  /*testCase = " Breeze 4096 * 4096 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.addmm(matrixLargeLeft, matrixLargeRight), testCase, 10)
  testCase = " Breeze 512 * 512 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixMidLeft.addmm(matrixMidLeft, matrixMidRight), testCase)
  testCase = " Breeze 32 * 32 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.addmm(matrixSmallLeft, matrixSmallRight), testCase)
  testCase = " Breeze 4096 * 4096 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorLarge.addmv(1, matrixLargeRight, vectorLarge), testCase, 10)
  testCase = " Breeze 512 * 512 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorMid.addmv(1, matrixMidRight, vectorMid), testCase)
  testCase = " Breeze 32 * 32 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorSmall.addmv(1, matrixSmallRight, vectorSmall), testCase)*/

  testCase = " Breeze 4096 * 4096 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixLargeRight,scalar), testCase, 1)

  testCase = " Breeze 512 * 512 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixMidRight, scalar), testCase)

  testCase = " Breeze 32 * 32 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixSmallRight, scalar), testCase)

  testCase = " Breeze 4096 * 4096 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixSmallRight), testCase)

  testCase = " Breeze 4096 * 4096 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixLargeRight), testCase, 1)

  testCase = " Breeze 512 * 512 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixMidRight), testCase)

  testCase = " Breeze 32 * 32 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixSmallRight), testCase)
}