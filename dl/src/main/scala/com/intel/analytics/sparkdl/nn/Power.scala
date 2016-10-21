/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Power[@specialized(Float, Double) T: ClassTag](
  val power: Int,
  val scale : Double = 1,
  val shift : Double = 0)
(implicit ev: TensorNumeric[T]) extends Module[T] {

  val diffScale = power * scale

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.copy(input)
    if(scale != 1) {
      output.mul(ev.fromType[Double](scale))
    }
    if(shift != 0) {
      output.add(ev.fromType[Double](shift))
    }
    if(power != 1) {
      output.pow(output, ev.fromType[Double](power))
    }

    output
  }


  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
    //               = diff_scale * y / (shift + scale * x)
    if(power == 2) {
      // Special case for y = (shift + scale * x)^2
      //     -> dy/dx = 2 * scale * (shift + scale * x)
      //              = diff_scale * shift + diff_scale * scale * x
      gradInput.copy(input)
      gradInput.mul(ev.fromType[Double](diffScale * shift))
      if(shift != 0) {
        gradInput.add(ev.fromType(diffScale * shift))
      }
    } else if (shift == 0) {
      // Special case for y = (scale * x)^power
      //     -> dy/dx = scale * power * (scale * x)^(power - 1)
      //              = scale * power * (scale * x)^power * (scale * x)^(-1)
      //              = power * y / x
      gradInput.fill(ev.fromType[Int](0))
      gradInput = output.addcdiv(ev.fromType[Double](power), output, input)
    } else {
      gradInput.copy(input)
      if(scale != 1) {
        gradInput.mul(ev.fromType[Double](scale))
      }
      if(shift != 0) {
        gradInput.add(ev.fromType[Double](shift))
      }
      gradInput.cdiv(gradInput, output)
      if (diffScale != 1) {
        gradInput.mul(ev.fromType[Double](diffScale))
      }
    }
    if(diffScale != 0) {
      gradInput.cmul(gradOutput)
    }

    gradInput
  }

}
