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


package com.tencent.angel.ml.metric


class LossMetric(var sampleNum: Int) extends Metric {
  var loss: Double = 0.0

  /**
    * 汇总各个Task的sampleNum、globalLoss值，汇总方式为累加。
    * @param other
    * @return
    */
  override def merge(other: this.type): this.type = {
    this.sampleNum += other.sampleNum
    this.loss += other.loss
    this
  }

  /**
    * 计算全局指标，计算方法为全局globalLoss值除样本数：
    * @return metric value
    */
  override def calculate(): Double = {
    this.loss / this.sampleNum
  }

  override def setValue(value: Double): Unit = {
    this.loss = value
  }

}

object LossMetric {
  def apply(sampleNum: Int): Metric = {
    new LossMetric(sampleNum)
  }
}
