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


package com.tencent.angel.worker.task;

import com.tencent.angel.exception.AngelException;

/**
 * The interface base task.
 *
 * @param <KEYIN>    the key type
 * @param <VALUEIN>  the value type
 * @param <VALUEOUT> the parsed value type
 */
public interface BaseTaskInterface<KEYIN, VALUEIN, VALUEOUT> {
    /**
     * 解析原始输入数据的一行，生成训练过程需要的数据结构
     *
     * @param key   the key
     * @param value the value
     * @return the valueout
     */
    VALUEOUT parse(KEYIN key, VALUEIN value);

    /**
     * 表示从原始数据块到训练数据集合的转换过程
     *
     * @param taskContext the task context
     */
    void preProcess(TaskContext taskContext);

    /**
     * 模型训练过程
     *
     * @param taskContext the task context
     * @throws Exception
     */
    void run(TaskContext taskContext) throws AngelException;
}
