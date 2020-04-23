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


package com.tencent.angel.ml.model

import java.util.concurrent.Future
import java.util.{ArrayList, List}

import com.tencent.angel.conf.MatrixConf
import com.tencent.angel.exception.{AngelException, InvalidParameterException}
import com.tencent.angel.ml.math2.utils.RowType
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.get.base.{GetFunc, GetResult}
import com.tencent.angel.ml.matrix.psf.update.base.{UpdateFunc, VoidResult}
import com.tencent.angel.ml.matrix.psf.update.zero.Zero
import com.tencent.angel.ml.matrix.psf.update.zero.Zero.ZeroParam
import com.tencent.angel.ml.matrix.{MatrixContext, MatrixOpLogType}
import com.tencent.angel.psagent.matrix.transport.adapter.{GetRowsResult, RowIndex}
import com.tencent.angel.worker.task.TaskContext
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.mutable.Map


/**
  * Angel 核心类
  * 提供了常用的远程矩阵（Matrix）和向量（Vector）的获取和更新接口，
  * 使得算法工程师可以如同操作本地对象一样的操作参数服务器上的**分布式矩阵和向量**
  *
  * @param modelName     模型名称
  * @param row           矩阵行数
  * @param col           矩阵列数
  * @param blockRow      每一个矩阵分片的行数
  * @param blockCol      每一个矩阵分片的列数
  * @param validIndexNum 有效索引数
  * @param needSave      是否需要保存文件系统
  * @param ctx           TaskContext PSModel的Task上下文 ,PSModel对象需要与Angel的一个Task绑定，
  *                      因为PSModel运行于Worker之上，另外也是为了支持BSP和SSP同步模型
  *                      这里使用了隐式转换，只要创建PSModel的容器中，有ctx这个对象，
  *                      它就会自动的将ctx注入PSModel之中，无需显式调用）
  */
class PSModel(
               val modelName: String,
               val row: Int,
               val col: Long,
               val blockRow: Int = -1,
               val blockCol: Long = -1,
               val validIndexNum: Long = -1,
               var needSave: Boolean = true)(implicit ctx: TaskContext) {

  val LOG: Log = LogFactory.getLog(classOf[PSModel])

  /** Matrix configuration */
  val matrixCtx = new MatrixContext(modelName, row, col, validIndexNum, blockRow, blockCol)

  /** Get task context */
  def getTaskContext = ctx

  /** Get matrix context */
  def getContext = matrixCtx

  /** Get ps matrix client */
  def getClient = ctx.getMatrix(modelName)

  // =======================================================================
  // Get and Set Area
  // =======================================================================


  /**
    * Get matrix id
    *
    * @return matrix id
    */
  def getMatrixId(): Int = {
    return getClient.getMatrixId
  }

  /**
    * Set model need to be saved
    *
    * @param _needSave
    * @return
    */
  def setNeedSave(_needSave: Boolean): this.type = {
    this.needSave = _needSave
    this
  }

  /**
    * 设置矩阵属性
    * Angel可以支持自定义矩阵参数扩展
    *
    * @param key   属性名
    * @param value 属性值
    */
  def setAttribute(key: String, value: String): this.type = {
    matrixCtx.set(key, value)
    this
  }

  /**
    * 设置平均属性
    * 设置矩阵更新属性，在更新矩阵时是否先将更新参数除以总的task数量。
    * 本属性increment函数中使用，但不影响update函数
    *
    * @param aver true表示在更新矩阵时先将更新参数除以总task数量，false表示不除
    */
  def setAverage(aver: Boolean): this.type = {
    matrixCtx.set(MatrixConf.MATRIX_AVERAGE, String.valueOf(aver))
    this
  }

  /**
    * 设置矩阵属性，是否采用hogwild 方式存储和更新本地矩阵。
    * 当worker task数量大于1时，可以使用hogwild 方式节省内存。默认为使用**hogwild** 方式
    *
    * @param hogwild true表示使用hogwild 方式，false表示不使用
    */
  def setHogwild(hogwild: Boolean): this.type = {
    matrixCtx.set(MatrixConf.MATRIX_HOGWILD, String.valueOf(hogwild))
    this
  }

  /**
    * 设置矩阵更新存储类型
    * 矩阵更新存储方式，当使用increment方法更新矩阵时，Angel会先将更新行向量缓存在本地。
    * 缓存的方式是在本地定义一个和待更新矩阵维度一致的矩阵
    *
    * @param oplogType storage type
    *                  - oplogType: String
    *                  目前支持的存储方式有：
    *                  DENSE_DOUBE： 表示使用一个稠密double型矩阵来存储矩阵的更新，一般当待更新矩阵元素类型为double时选择更新存储方式
    *                  DENSE_INT：   表示使用一个稠密的int型矩阵来存储矩阵更新，一般当待更新矩阵元素为int类型时选择这种存储方式
    *                  LIL_INT：     表示使用一个稀疏int型矩阵来存储矩阵更新，当待更新矩阵元素类型为int且更新相对稀疏时选择更新存储方式
    *                  DENSE_FLOAT： 表示使用一个稠密的float型矩阵来存储矩阵更新，一般当待更新矩阵元素为float类型时选择这种存储方式
    */
  def setOplogType(oplogType: String): this.type = {
    matrixCtx.set(MatrixConf.MATRIX_OPLOG_TYPE, oplogType)
    this
  }

  /**
    * Set the matrix update storage type
    *
    * @param oplogType storage type
    */
  def setOplogType(oplogType: MatrixOpLogType): this.type = {
    matrixCtx.set(MatrixConf.MATRIX_OPLOG_TYPE, oplogType.name())
    this
  }

  /**
    * 设置矩阵行类型
    * 设置矩阵行向量的元素类型和存储方式，可以根据模型特点和稀疏程度来设置该参数。
    * 目前Angel支持的矩阵元素类型有int, float和double；存储方式有稀疏和稠密
    *
    * @param rowType MLProtos.RowType
    *                目前支持矩阵行向量的元素类型和存储方式有：
    *                T_DOUBLE_SPARSE： 表示稀疏double型
    *                T_DOUBLE_DENSE ： 表示稠密double型,
    *                T_INT_SPARSE：    表示稀疏int型；
    *                T_INT_DENSE：     表示稠密int型；
    *                T_FLOAT_SPARSE：  表示稀疏float型；
    *                T_FLOAT_DENSE：   表示稠密float型；
    *                T_INT_ARBITRARY ：表示数据类型为int
    *                用户可以根据实际算法情况，选择最节省内存的存储方式
    */
  def setRowType(rowType: RowType): this.type = {
    matrixCtx.setRowType(rowType)
    this
  }

  def getRowType(): RowType = matrixCtx.getRowType

  /**
    * 设置模型（矩阵）加载路径，本属性用于模型增量更新和预测功能模式下，
    * 表示在参数服务器端初始化矩阵时，从文件中读取矩阵参数来初始化
    *
    * @param path load path
    */
  def setLoadPath(path: String): this.type = {
    matrixCtx.set(MatrixConf.MATRIX_LOAD_PATH, path)
    LOG.info("Before training, matrix " + this.matrixCtx.getName + " will be loaded from " + path)
    this
  }

  /**
    * 设置模型(矩阵)保存路径；在训练功能模式下，当训练完成时，
    * 需要将参数服务器上的矩阵参数保存在文件中
    *
    * @param path
    */
  def setSavePath(path: String): this.type = {
    matrixCtx.set(MatrixConf.MATRIX_SAVE_PATH, path)
    LOG.info("After training matrix " + this.matrixCtx.getName + " will be saved to " + path)
    this
  }

  // =======================================================================
  // Sync Area
  // =======================================================================

  /**
    * 默认clock的简化版，封装了clock().get()。除非需要的等待，否则建议调用改方法
    */
  def syncClock(flush: Boolean = true) = {
    this.clock(flush).get()
  }

  /**
    * 将本地缓存的所有矩阵更新（调用increment函数会将更新缓存在本地）合并后发送给参数服务器，然后更新矩阵的时钟
    *
    * @param flush 是否刷新缓存 oplog
    * @throws com.tencent.angel.exception.AngelException
    * @return Future[VoidResult] clock操作结果，应用程序可以选择是否等待clock操作完成
    */
  @throws(classOf[AngelException])
  def clock(flush: Boolean = true): Future[VoidResult] = {
    try {
      return getClient.clock(flush)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }


  /**
    * 将本地缓存的所有矩阵更新（调用increment函数会将更新缓存在本地）合并后发送给参数服务器
    *
    * @throws com.tencent.angel.exception.AngelException
    * @return Future[VoidResult] flush操作结果，应用程序可以选择是否等待flush操作完成
    */
  @throws(classOf[AngelException])
  def flush(): Future[VoidResult] = {
    try {
      return getClient.flush
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }

  // =======================================================================
  // Remote Model Area
  // =======================================================================


  /**
    * 以累加的方式更新矩阵的某一行，该方法采用异步更新的方式，只是将更新向量缓存到本地，
    * 而非直接作用于参数服务器，只有当执行flush或者clock方法时才会将更新作用到参数服务器
    *
    * @param delta delta Vector更新行向量，Vector 与行向量维度一致的更新向量
    * @throws com.tencent.angel.exception.AngelException
    */
  @throws(classOf[AngelException])
  def increment(delta: Vector) {
    try {
      getClient.increment(delta)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }

  /**
    * Increment the matrix row vector use a same dimension vector. The update will be cache in local
    * and send to ps until flush or clock is called
    *
    * @param rowIndex row index
    * @param delta    update row vector
    * @throws com.tencent.angel.exception.AngelException
    */
  @throws(classOf[AngelException])
  def increment(rowIndex: Int, delta: Vector) {
    try {
      getClient.increment(rowIndex, delta)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }

  /**
    * 以累加的方式更新矩阵的某些行，该方法采用异步更新的方式，只是将更新向量缓存到本地，
    * 而非直接作用于参数服务器，只有当执行flush或者clock方法时才会将更新作用到参数服务器
    *
    * @param deltas List[Vector] 与行向量维度一致的更新向量列表
    * @throws com.tencent.angel.exception.AngelException
    */
  @throws(classOf[AngelException])
  def increment(deltas: List[Vector]) {
    import scala.collection.JavaConversions._
    for (delta <- deltas) increment(delta)
  }

  /**
    * 使用psf get函数获取矩阵的元素或元素统计信息。与getRow/getRows/getRowsFlow方法不同，本方法只支持异步模型
    *
    * @param func GetFunc get类型的psf函数。psf函数是Angel提供的一种参数服务器功能扩展接口
    * @throws com.tencent.angel.exception.AngelException
    * @return psf get函数返回结果
    */
  @throws(classOf[AngelException])
  def get(func: GetFunc): GetResult = {
    try {
      return getClient.get(func)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }


  /**
    * 获取矩阵的某一行。在不同的同步模型下，本方法会有不同的流程。
    * Angel支持3种同步模型：BSP,SSP和异步模型。
    *
    * 在BSP 和 SSP 模型下，本方法会首先检查本地缓存中是否已经存在需要获取的行且该行的时钟
    * 信息是否满足同步模型，若缓存中不存在或者不满足同步模型要求，它会向参数服务器请求，如
    * 果参数服务器上的行时钟也不满足同步模型，则它会一直等待直到满足为止；
    * 在异步模型下，该方法会直接向参数服务器请求所需的行，而不关心时钟信息
    *
    * @param rowIndex 行号
    * @throws com.tencent.angel.exception.AngelException
    * @return 指定行的向量
    */
  @SuppressWarnings(Array("unchecked"))
  @throws(classOf[AngelException])
  def getRow(rowIndex: Int): Vector = {
    try {
      return getClient.getRow(rowIndex)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }


  /**
    * 获取矩阵的某些行。在BSP/SSP/异步模型下的获取流程与getRow方法类似
    *
    * @param rowIndex 行号集合
    * @param batchNum the number of rows get in a rpc
    * @throws com.tencent.angel.exception.AngelException
    * @return row index to row map
    */
  @throws(classOf[AngelException])
  def getRows(rowIndex: RowIndex, batchNum: Int): Map[Int, Vector] = {
    val indexToVectorMap = scala.collection.mutable.Map[Int, Vector]()
    val rows = getRowsFlow(rowIndex, batchNum)
    try {
      var finish = false
      while (!finish) {
        rows.take() match {
          case null => finish = true
          case row => indexToVectorMap += (row.getRowId -> row)
        }
      }
    }
    catch {
      case e: Exception => {
        throw new AngelException(e)
      }
    }
    indexToVectorMap
  }

  /**
    * 获取矩阵的某些行。在BSP/SSP/异步模型下的获取流程与getRow方法类似
    *
    * @param rowIndexes 行号数组
    * @throws com.tencent.angel.exception.AngelException
    * @return 指定行的行向量列表；列表有序，与参数数组顺序一致
    */
  @throws(classOf[AngelException])
  def getRows(rowIndexes: Array[Int]): List[Vector] = {
    val rowIndex = new RowIndex()
    for (index <- rowIndexes) {
      rowIndex.addRowId(index)
    }

    val indexToVectorMap = getRows(rowIndex, -1)

    val rowList = new ArrayList[Vector](rowIndexes.length)

    for (i <- 0 until rowIndexes.length)
      rowList.add(indexToVectorMap.get(rowIndexes(i)).get)

    rowList
  }

  /**
    * 以流水化的形式获取矩阵的某些行，该方法会立即返回，用于支持一边计算一边进行行获取，
    * 在BSP/SSP/异步模型下的获取流程与getRow方法类似
    *
    * @param rowIndex 行号集合
    * @param batchNum 一次rpc请求获取行数，每次RPC请求的行数，这个参数定义了流水化的粒度，可以设置为-1，表示由系统自行选择
    * @throws com.tencent.angel.exception.AngelException
    * @return 一个行向量队列，上层应用程序可以从该队列中得到已经获取到的行对应的行向量，
    *         行结果，里面包含一个LinkedBlockingQueue<Vector>
    */
  @throws(classOf[AngelException])
  def getRowsFlow(rowIndex: RowIndex, batchNum: Int): GetRowsResult = {
    try {
      return getClient.getRowsFlow(rowIndex, batchNum)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }

  /**
    * 获取模型指定行的指定index对应的部分，用于32 bit的模型s
    *
    * @param rowId 行号
    * @param index 列下标数组
    * @return 一个稀疏类型的向量,Vector的key指定了一个索引数组
    */
  def getRowWithIndex(rowId: Int, index: Array[Int]): Vector = {
    getClient.get(rowId, index)
  }

  /**
    * 获取模型指定行的指定index对应的部分， 用于64 bit的模型
    *
    * @param rowId 行号
    * @param index 列下标数组
    * @return 一个稀疏类型的向量
    */
  def getRowWithLongIndex(rowId: Int, index: Array[Long]): Vector = {
    getClient.get(rowId, index)
  }

  /**
    * 使用psf update函数更新矩阵的参数。与increment方法不同，本方法会直接将更新作用与参数服务器端
    *
    * @param func func: GetFunc psf update函数。
    *             用户可以根据需求自定义psf update函数，当然，Angel提供了一个包含常用函数的函数库。
    *             与increment函数不同，本方法会立即将更新作用于参数服务器
    * @throws com.tencent.angel.exception.AngelException
    * @return Future[VoidResult] psf update函数返回结果，应用程序可以选择是否等待更新结果
    */
  @throws(classOf[AngelException])
  def update(func: UpdateFunc): Future[VoidResult] = {
    try {
      return getClient.asyncUpdate(func)
    }
    catch {
      case e: InvalidParameterException => {
        throw new AngelException(e)
      }
    }
  }

  /**
    * Set all matrix elements to zero
    *
    * @throws com.tencent.angel.exception.AngelException
    */
  @throws(classOf[AngelException])
  def zero() {
    val updater: Zero = new Zero(new ZeroParam(getMatrixId, false))
    try {
      update(updater).get
    }
    catch {
      case e: Any => {
        throw new AngelException(e)
      }
    }
  }

  override def finalize(): Unit = super.finalize()

}

object PSModel {
  def apply(modelName: String, row: Int, col: Long, blockRow: Int = -1, blockCol: Long = -1, nnz: Long = -1)(implicit ctx: TaskContext) = {
    new PSModel(modelName, row, col, blockRow, blockCol, nnz)(ctx)
  }
}
