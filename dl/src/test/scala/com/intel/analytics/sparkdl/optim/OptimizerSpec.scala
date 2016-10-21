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

package com.intel.analytics.sparkdl.optim

import com.intel.analytics.sparkdl.models.imagenet.AlexNet
import com.intel.analytics.sparkdl.nn.{Module, Sequential}
import com.intel.analytics.sparkdl.utils.{File, T, Table}
import org.scalatest.{FlatSpec, Matchers}

class OptimizerSpec extends FlatSpec with Matchers {
  val model = new Sequential[Float]()

  "Optimizer" should "end with maxEpoch" in {
    val dummyOptimizer = new Optimizer[Float](model, Trigger.maxEpoch(10)) {
      override def optimize(): Module[Float] = {
        val state = T("epoch" -> 9)
        endWhen(state) should be(false)
        state("epoch") = 10
        endWhen(state) should be(false)
        state("epoch") = 11
        endWhen(state) should be(true)
        model
      }
    }
    dummyOptimizer.optimize()
  }

  it should "end with iteration" in {
    val dummyOptimizer = new Optimizer[Float](model, Trigger.maxIteration(1000)) {
      override def optimize(): Module[Float] = {
        val state = T("neval" -> 999)
        endWhen(state) should be(false)
        state("neval") = 1000
        endWhen(state) should be(false)
        state("neval") = 1001
        endWhen(state) should be(true)
        model
      }
    }
    dummyOptimizer.optimize()
  }

  it should "be triggered every epoch" in {
    val dummyOptimizer = new Optimizer[Float](model, Trigger.maxEpoch(10)) {
      override def optimize(): Module[Float] = {
        val state = T("epoch" -> 9)
        validationTrigger.get(state) should be(false)
        cacheTrigger.get(state) should be(false)
        state("epoch") = 10
        validationTrigger.get(state) should be(true)
        cacheTrigger.get(state) should be(true)
        validationTrigger.get(state) should be(false)
        cacheTrigger.get(state) should be(false)
        state("epoch") = 11
        validationTrigger.get(state) should be(true)
        cacheTrigger.get(state) should be(true)
        cachePath.isDefined should be(true)
        model
      }
    }
    dummyOptimizer.setValidationTrigger(Trigger.everyEpoch)
    dummyOptimizer.setCache("", Trigger.everyEpoch)
    dummyOptimizer.optimize()
  }

  it should "be triggered every 5 iterations" in {
    val dummyOptimizer = new Optimizer[Float](model, Trigger.maxEpoch(5)) {
      override def optimize(): Module[Float] = {
        val state = T("neval" -> 1)
        validationTrigger.get(state) should be(false)
        cacheTrigger.get(state) should be(false)
        state("neval") = 4
        validationTrigger.get(state) should be(false)
        cacheTrigger.get(state) should be(false)
        state("neval") = 5
        validationTrigger.get(state) should be(true)
        cacheTrigger.get(state) should be(true)
        model
      }
    }
    dummyOptimizer.setValidationTrigger(Trigger.severalIteration(5))
    dummyOptimizer.setCache("", Trigger.severalIteration(5))
    dummyOptimizer.optimize()
  }

  it should "save model to given path" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    val model = AlexNet[Float](1000)
    val dummyOptimizer = new Optimizer[Float](model, Trigger.severalIteration(5)) {
      override def optimize(): Module[Float] = {
        saveModel()
        model
      }
    }
    dummyOptimizer.setCache(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    val loadedModel = File.loadObj[Module[Double]](filePath + ".model")
    loadedModel should be(model)
  }

  it should "save model and state to given path with postfix" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "model").getAbsolutePath
    val model = AlexNet[Float](1000)
    val dummyOptimizer = new Optimizer[Float](model, Trigger.severalIteration(5)) {
      override def optimize(): Module[Float] = {
        saveModel(".test")
        model
      }
    }
    dummyOptimizer.setCache(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    val loadedModel = File.loadObj[Module[Double]](filePath + ".model.test")
    loadedModel should be(model)
  }

  it should "save state to given path" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "state").getAbsolutePath
    val state = T("test" -> 123)
    val dummyOptimizer = new Optimizer[Float](model, Trigger.severalIteration(5)) {
      override def optimize(): Module[Float] = {
        saveState(state)
        model
      }
    }
    dummyOptimizer.setCache(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    val loadedState = File.loadObj[Table](filePath + ".state")
    loadedState should be(state)
  }

  it should "save state to given path with post fix" in {
    val filePath = java.io.File.createTempFile("OptimizerSpec", "state").getAbsolutePath
    val state = T("test" -> 123)
    val dummyOptimizer = new Optimizer[Float](model, Trigger.severalIteration(5)) {
      override def optimize(): Module[Float] = {
        saveState(state, ".post")
        model
      }
    }
    dummyOptimizer.setCache(filePath, Trigger.everyEpoch)
    dummyOptimizer.optimize()

    val loadedState = File.loadObj[Table](filePath + ".state.post")
    loadedState should be(state)
  }
}
