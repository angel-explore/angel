package com.tencent.angel.ml;

import com.tencent.angel.utils.HdfsUtil;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.compress.CompressionOutputStream;
import org.apache.hadoop.io.compress.GzipCodec;

/**
 * User:krisjin
 * Date:2020-05-22
 */
public class ModelUpload {


    public static void out(String modelPath, Configuration conf) {

        Class<?> codecClass = null;
        try {
            GzipCodec codec = new GzipCodec();
            codecClass = Class.forName("org.apache.hadoop.io.compress.GzipCodec");

            FileSystem fs = FileSystem.get(conf);
//            CompressionCodec codec = (CompressionCodec) ReflectionUtils.newInstance(codecClass, conf);
            //指定压缩文件路径
            FSDataOutputStream outputStream = fs.create(new Path("/usr/local/tools/ziptest/lr.tz"));
            //指定要被压缩的文件路径
            FSDataInputStream in = fs.open(new Path(modelPath));
            //创建压缩输出流
            CompressionOutputStream out = codec.createOutputStream(outputStream);
            IOUtils.copyBytes(in, out, conf);
            IOUtils.closeStream(in);
            IOUtils.closeStream(out);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void uploadOSS(String modelPath, Configuration conf) {
        try {
            String local = "/usr/local/tools/ziptest/";
            String modelName = modelPath.substring(modelPath.lastIndexOf("/") + 1);

            FileSystem fs = FileSystem.get(conf);

            String newName = local + modelName;
            HdfsUtil.copyFilesInSameHdfs(new Path(modelPath), new Path(newName), fs);

            FileZipUtil.zipCompress(newName, newName + ".zip");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
