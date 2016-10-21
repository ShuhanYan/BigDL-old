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

package com.intel.analytics.sparkdl.models.imagenet

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object Vgg_16 {
  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val model = new Sequential[T]()
    model.add(new SpatialConvolution[T](3, 64, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](64, 64, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](64, 128, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](128, 128, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](128, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](256, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new View[T](512 * 7 * 7))
    model.add(new Linear[T](512 * 7 * 7, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](4096, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](4096, classNum))
    model.add(new LogSoftMax[T]())

    model
  }
}

object Vgg_19 {
  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val model = new Sequential[T]()
    model.add(new SpatialConvolution[T](3, 64, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](64, 64, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](64, 128, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](128, 128, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](128, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](256, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialConvolution[T](512, 512, 3, 3, 1, 1, 1, 1))
    model.add(new ReLU[T](true))
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))

    model.add(new View[T](512 * 7 * 7))
    model.add(new Linear[T](512 * 7 * 7, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](4096, 4096))
    model.add(new Threshold[T](0, 1e-6))
    model.add(new Dropout[T](0.5))
    model.add(new Linear[T](4096, classNum))
    model.add(new LogSoftMax[T]())

    model
  }
}
