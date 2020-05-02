/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */


package com.tencent.angel.worker.task

import java.io.IOException

import com.tencent.angel.exception.AngelException
import com.tencent.angel.ml.math2.utils.{DataBlock, LabeledData}
import com.tencent.angel.ml.model.OldMLModel
import com.tencent.angel.utils.HdfsUtil

abstract class PredictTask[KEYIN, VALUEIN](ctx: TaskContext)
  extends BaseTask[KEYIN, VALUEIN, LabeledData](ctx) {

  /**
    * 利用模型进行预测
    *
    * @param taskContext the task context
    * @throws com.tencent.angel.exception.AngelException
    */
  @throws(classOf[AngelException])
  final def run(taskContext: TaskContext) {
    this.predict(taskContext)
  }

  def predict(taskContext: TaskContext)

  @throws(classOf[IOException])
  protected final def predict(
                               taskContext: TaskContext,
                               model: OldMLModel,
                               dataBlock: DataBlock[LabeledData]) {
    val predictResult = model.predict(dataBlock)
    HdfsUtil.writeStorage(predictResult, taskContext)
  }
}
