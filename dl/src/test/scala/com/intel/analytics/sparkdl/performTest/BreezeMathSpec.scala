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
  var matrixLargeResult = new DenseMatrix[Float](sizeLarge, sizeLarge)
  val vectorLarge = DenseVector.rand(sizeLarge ,Rand.uniform.map(_.toFloat))
  val sizeMid = 512
  val matrixMidLeft = init(sizeMid,-1000,1000,0.5f)
  val matrixMidRight = init(sizeMid,-1000,1000,0.5f)
  var matrixMidResult = new DenseMatrix[Float](sizeMid,sizeMid)
  val vectorMid = DenseVector.rand(sizeMid ,Rand.uniform.map(_.toFloat))
  val sizeSmall = 32
  val matrixSmallLeft = init(sizeSmall,-500,500,1)
  val matrixSmallRight = init(sizeSmall,-500,500,1)
  var matrixSmallResult = new DenseMatrix[Float](sizeSmall,sizeSmall)
  val vectorSmall = DenseVector.rand(sizeSmall ,Rand.uniform.map(_.toFloat))
  val scalar = 5

  var testCase = " Breeze 4096 * 4096 matrix add operation"
  TestUtils.testMathOperation(() => matrixLargeResult = matrixLargeLeft :+ matrixLargeRight, testCase)

  testCase = " Breeze 512 * 512 matrix add operation"
  TestUtils.testMathOperation(() => matrixMidLeft:+matrixMidRight, testCase, 100)

  testCase = " Breeze 32 * 32 matrix add operation"
  TestUtils.testMathOperation(() => matrixSmallResult = matrixSmallLeft:+matrixSmallRight, testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix minus operation"
  TestUtils.testMathOperation(() => matrixLargeLeft:-matrixLargeRight, testCase)

  testCase = " Breeze 512 * 512 matrix minus operation"
  TestUtils.testMathOperation(() => matrixMidLeft:-matrixMidRight, testCase, 100)

  testCase = " Breeze 32 * 32 matrix minus operation"
  TestUtils.testMathOperation(() => matrixSmallResult = matrixSmallLeft:-matrixSmallRight, testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixLargeLeft:*matrixLargeRight, testCase)

  testCase = " Breeze 512 * 512 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixMidLeft:*matrixMidRight, testCase, 100)

  testCase = " Breeze 32 * 32 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixSmallResult = matrixSmallLeft :* matrixSmallRight, testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix divide operation"
  TestUtils.testMathOperation(() => matrixLargeLeft:/matrixLargeRight, testCase)

  testCase = " Breeze 512 * 512 matrix divide operation"
  TestUtils.testMathOperation(() => matrixMidLeft:/matrixMidRight, testCase, 100)

  testCase = " Breeze 32 * 32 matrix divide operation"
  TestUtils.testMathOperation(() => matrixSmallResult = matrixSmallLeft:/matrixSmallRight, testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixLargeResult = matrixLargeLeft * matrixLargeRight, testCase)

  testCase = " Breeze 512 * 512 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixMidResult = matrixMidLeft * matrixMidRight, testCase, 100)

  testCase = " Breeze 32 * 32 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixSmallResult = matrixSmallLeft * matrixSmallRight, testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix addmv operation"
  TestUtils.testMathOperation(() => matrixLargeRight * vectorLarge, testCase, 10)

  testCase = " Breeze 512 * 512 matrix addmv operation"
  TestUtils.testMathOperation(() => matrixMidRight * vectorMid, testCase, 100)

  testCase = " Breeze 32 * 32 matrix addmv operation"
  TestUtils.testMathOperation(() => matrixSmallRight * vectorSmall, testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixLargeRight,scalar), testCase)

  testCase = " Breeze 512 * 512 matrix pow operation"
  TestUtils.testMathOperation(() => pow(matrixMidRight, scalar), testCase, 100)

  testCase = " Breeze 32 * 32 matrix pow operation"
  TestUtils.testMathOperation(() => matrixSmallResult = pow(matrixSmallRight, scalar), testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixLargeRight), testCase)

  testCase = " Breeze 512 * 512 matrix log operation"
  TestUtils.testMathOperation(() => log(matrixMidRight), testCase, 100)

  testCase = " Breeze 32 * 32 matrix log operation"
  TestUtils.testMathOperation(() => matrixSmallResult = log(matrixSmallRight), testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixLargeRight), testCase)

  testCase = " Breeze 512 * 512 matrix exp operation"
  TestUtils.testMathOperation(() => exp(matrixMidRight), testCase, 100)

  testCase = " Breeze 32 * 32 matrix exp operation"
  TestUtils.testMathOperation(() => matrixSmallResult = exp(matrixSmallRight), testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixLargeRight), testCase)

  testCase = " Breeze 512 * 512 matrix sqrt operation"
  TestUtils.testMathOperation(() => sqrt(matrixMidRight), testCase, 100)

  testCase = " Breeze 32 * 32 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixSmallResult = sqrt(matrixSmallRight), testCase, 1000)

  testCase = " Breeze 4096 * 4096 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixLargeRight), testCase)

  testCase = " Breeze 512 * 512 matrix log1p operation"
  TestUtils.testMathOperation(() => log1p(matrixMidRight), testCase, 100)

  testCase = " Breeze 32 * 32 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixSmallResult = log1p(matrixSmallRight), testCase, 1000)
}