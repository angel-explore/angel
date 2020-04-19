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


package com.tencent.angel.client;

import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.matrix.MatrixContext;
import com.tencent.angel.ml.model.OldMLModel;
import com.tencent.angel.model.ModelLoadContext;
import com.tencent.angel.model.ModelSaveContext;
import com.tencent.angel.worker.task.BaseTask;

import java.util.List;

/**
 * Angel client interface. It defines application control operations from angel client.
 */
public interface AngelClientInterface {

  /**
   * Add a new matrix.
   *
   * @param mContext matrix context
   * @throws AngelException
   */
  void addMatrix(MatrixContext mContext) throws AngelException;

  /**
   * Add a new matrix.
   *
   * @throws AngelException
   */
  void createMatrices() throws AngelException;

  /**
   * Add a new matrix.
   *
   * @throws AngelException
   */
  void createMatrices(List<MatrixContext> contexts) throws AngelException;

  /**
   * 启动Angel 参数服务器；该方法会首先启动Angel Master，
   * 然后向Angel Master发送启动参数服务器的命令，
   * 直到所有的参数服务器启动成功该方法才会返回
   * @throws AngelException
   */
  void startPSServer() throws AngelException;

  /**
   * 从文件加载模型。该方法在不同的功能模式下有不同的流程，
   * 目前Angel支持训练 和预测 两种功能模式，
   * 其中训练又可分为新模型训练和模型增量更新两种方式。在模型增量更新 和预测 模式下，
   * 该方法会首先将已经训练好的模型元数据和参数加载到参数服务器的内存中，为下一步计算做准备；
   * 在新模型训练 模式下，该方法直接在参数服务器中定义新的模型并初始化模型参数
   * @param model model
   * @throws AngelException
   */
  void loadModel(OldMLModel model) throws AngelException;

  /**
   * Load the model from files
   *
   * @param context model load context
   * @throws AngelException
   */
  void load(ModelLoadContext context) throws AngelException;

  /**
   * Recover the model from the checkpoint
   * @param checkpointId the checkpoint id
   * @param context load context
   * @throws AngelException
   */
  void recover(int checkpointId, ModelLoadContext context) throws AngelException;

  /**
   * 接收任务并启动
   * 启动计算过程。该方法启动Worker和Task，开始执行具体的计算任务
   * @param taskClass  taskClass：Task计算流程，一般情况下Angel mllib中的每一种算法都提供了对应的Task实现，
   *                   但当用户需要修改具体执行流程或者实现新的算法时，需要提供一个自定义的实现
   * @throws AngelException
   */
  void runTask(@SuppressWarnings("rawtypes") Class<? extends BaseTask> taskClass)
    throws AngelException;

  /**
   * Startup workers and start to execute tasks.
   * <p>
   * Use #runTask instead
   *
   * @throws AngelException
   */
  @Deprecated void run() throws AngelException;

  /**
   * Wait until all the tasks are done.
   * 等待直到所有task计算完成
   * 该方法在训练和预测功能模式下有不同的流程。
   * 在训练模式下，具体的训练计算过程一旦完成，该方法就会返回
   * 在预测模式下，在预测计算过程完成后，该方法会将保存在临时输出目录下的预测结果rename到最终输出目录
   * @throws AngelException
   */
  void waitForCompletion() throws AngelException;

  /**
   * 保存计算好的模型，存储hdfs。
   * 该方法只有在训练模式下才会有效
   *
   * @param model model need to write to files.
   * @throws AngelException
   */
  void saveModel(OldMLModel model) throws AngelException;

  /**
   * Save the model to files
   *
   * @param context model save context
   * @throws AngelException
   */
  void save(ModelSaveContext context) throws AngelException;

  /**
   * Write the model checkpoint
   * @param checkpointId the checkpoint id
   * @param context save context
   * @throws AngelException
   */
  void checkpoint(int checkpointId, ModelSaveContext context) throws AngelException;

  /**
   * Stop the whole application.
   *
   * @throws AngelException stop failed
   */
  void stop() throws AngelException;

  /**
   * 在给定状态下，停止angel应用
   * 结束任务，释放计算资源，清理临时目录等
   * @param stateCode 0:succeed,1:killed,2:failed
   * @throws AngelException stop failed
   */
  void stop(int stateCode) throws AngelException;

  /**
   * Kill the application
   *
   * @throws AngelException
   */
  void kill() throws AngelException;

  /**
   * Clean thre resource
   *
   * @throws AngelException
   */
  void close() throws AngelException;
}
