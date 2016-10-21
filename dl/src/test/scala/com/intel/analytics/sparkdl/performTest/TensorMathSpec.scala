package com.intel.analytics.sparkdl.performTest

import com.intel.analytics.sparkdl.tensor._
import org.scalatest.FlatSpec
import com.intel.analytics.sparkdl.utils.RandomGenerator._

/**
 * Created by yao on 9/7/16.
 */
class TensorMathSpec extends FlatSpec {
  def init(length:Int,rangeMin:Float,rangeMax:Float,interval:Float): Tensor[Float] ={
    val result = Tensor[Float](length, length).fill(1.0f)
    var const = rangeMin
    for (i <- 1 until(length+1)){
      for (j <- 1 until(length+1)) {
        if (const > rangeMax) const = rangeMin
        result.setValue(i, j, const)
        const = const+interval
      }
    }
    result
  }
  val Seed = 100
  RNG.setSeed(Seed)
  val sizeLarge = 4096
  val matrixLargeLeft = init(sizeLarge,-1000,1000,0.5f)
  val matrixLargeRight = init(sizeLarge,-500,500,0.5f)
  val matrixLargeResult = Tensor[Float](sizeLarge, sizeLarge).fill(0.0f)
  val vectorLarge = Tensor[Float](sizeLarge).rand()
  val sizeMid = 512
  val matrixMidLeft = init(sizeMid,-1000,1000,0.5f)
  val matrixMidRight = init(sizeMid,-1000,1000,0.5f)
  val matrixMidResult = Tensor[Float](sizeMid, sizeMid).fill(0.0f)
  val vectorMid = Tensor[Float](sizeMid).rand()
  val sizeSmall = 32
  val matrixSmallLeft = init(sizeSmall,-500,500,1)
  val matrixSmallRight = init(sizeSmall,-500,500,1)
  val matrixSmallResult = Tensor[Float](sizeSmall, sizeSmall).fill(0.0f)
  val vectorSmall = Tensor[Float](sizeSmall).rand()
  val scalar = 5

  var testCase = "4096 * 4096 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixLargeResult.addmm(matrixLargeLeft, matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixMidResult.addmm(matrixMidLeft, matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix addmm operation"
  TestUtils.testMathOperation(() => matrixSmallResult.addmm(matrixSmallLeft, matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorLarge.addmv(1, matrixLargeRight, vectorLarge), testCase, 100)

  testCase = "512 * 512 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorMid.addmv(1, matrixMidRight, vectorMid), testCase, 1000)

  testCase = "32 * 32 matrix addmv operation"
  TestUtils.testMathOperation(() => vectorSmall.addmv(1, matrixSmallRight, vectorSmall), testCase, 100000)

  testCase = "4096 * 4096 matrix add operation"
  TestUtils.testMathOperation(() => matrixLargeResult.add(matrixLargeLeft,matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix add operation"
  TestUtils.testMathOperation(() => matrixMidResult.add(matrixMidLeft,matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix add operation"
  TestUtils.testMathOperation(() => matrixSmallResult.add(matrixSmallLeft,matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix minus operation"
  TestUtils.testMathOperation(() => matrixLargeResult.sub(matrixLargeLeft,matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix minus operation"
  TestUtils.testMathOperation(() => matrixMidResult.sub(matrixMidLeft,matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix minus operation"
  TestUtils.testMathOperation(() => matrixSmallResult.sub(matrixSmallLeft,matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixLargeResult.cmul(matrixLargeLeft,matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixMidResult.cmul(matrixMidLeft,matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix multiply operation"
  TestUtils.testMathOperation(() => matrixSmallResult.cmul(matrixSmallLeft,matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix divide operation"
  TestUtils.testMathOperation(() => matrixLargeResult.cdiv(matrixLargeLeft,matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix divide operation"
  TestUtils.testMathOperation(() => matrixMidResult.cdiv(matrixMidLeft,matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix divide operation"
  TestUtils.testMathOperation(() => matrixSmallResult.cdiv(matrixSmallLeft,matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix pow operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.pow(matrixLargeRight,scalar), testCase, 100)

  testCase = "512 * 512 matrix pow operation"
  TestUtils.testMathOperation(() => matrixMidLeft.pow(matrixMidRight, scalar), testCase, 1000)

  testCase = "32 * 32 matrix pow operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.pow(matrixSmallRight, scalar), testCase, 100000)

  testCase = "4096 * 4096 matrix log operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.log(matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix log operation"
  TestUtils.testMathOperation(() => matrixMidLeft.log(matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix log operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.log(matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix exp operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.exp(matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix exp operation"
  TestUtils.testMathOperation(() => matrixMidLeft.exp(matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix exp operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.exp(matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.sqrt(matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixMidLeft.sqrt(matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix sqrt operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.sqrt(matrixSmallRight), testCase, 100000)

  testCase = "4096 * 4096 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixLargeLeft.log1p(matrixLargeRight), testCase, 100)

  testCase = "512 * 512 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixMidLeft.log1p(matrixMidRight), testCase, 1000)

  testCase = "32 * 32 matrix log1p operation"
  TestUtils.testMathOperation(() => matrixSmallLeft.log1p(matrixSmallRight), testCase, 100000)

}