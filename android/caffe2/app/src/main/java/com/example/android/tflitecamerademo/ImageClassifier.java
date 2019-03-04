/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
//import org.tensorflow.lite.Interpreter;
import caffe2.Caffe2;
import caffe2.PredictorData;


/** Classifies images with Tensorflow Lite. */
public class ImageClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteCameraDemo";

  /** Name of the model file stored in Assets. */
  private static final String INIT_MODEL_PATH = "init_net.pb";
  private static final String PREDICT_MODEL_PATH = "predict_net.pb";

  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "labels.txt";
  public String LABEL_PATH_LOCAL;

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  static final int DIM_IMG_SIZE_X = 224;
  static final int DIM_IMG_SIZE_Y = 224;

  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128.0f;


  /* Preallocated buffers for storing image data in. */
  private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  //private Interpreter tflite;
  private PredictorData mypredictor;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  private ByteBuffer imgData = null;

  /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
  private float[][] labelProbArray = null;
  /** multi-stage low pass filter **/
  private float[][] filterLabelProbArray = null;
  private static final int FILTER_STAGES = 3;
  private static final float FILTER_FACTOR = 0.4f;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });

  /* DEMO: Initializes an {@code ImageClassifier}.
  ImageClassifier(Activity activity) throws IOException {
    tflite = new Interpreter(loadModelFile(activity));
    labelList = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
    labelProbArray = new float[1][labelList.size()];
    filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }*/

  ImageClassifier(Activity activity) throws IOException {
    try{

      // TRY temporary storage
      AssetManager assetManager = activity.getAssets();
      String abi = Build.CPU_ABI;
      String filesDir = activity.getFilesDir().getPath();
      String testPathInit = abi + "/" + INIT_MODEL_PATH;
      String testPathPredict = abi + "/" + PREDICT_MODEL_PATH;
      String testPathLabels = abi + "/" + LABEL_PATH;

      InputStream inStreamInit = assetManager.open(INIT_MODEL_PATH);
      Log.d(TAG, "Opened" + INIT_MODEL_PATH);
      InputStream inStreamPredict = assetManager.open(PREDICT_MODEL_PATH);
      Log.d(TAG, "Opened" + PREDICT_MODEL_PATH);
      InputStream inStreamLabels = assetManager.open(LABEL_PATH);
      Log.d(TAG, "Opened" + LABEL_PATH);

      // Copy this file to an executable location
      File outFileInit = new File(filesDir, INIT_MODEL_PATH);
      File outFilePredict = new File(filesDir, PREDICT_MODEL_PATH);
      File outFileLabels = new File(filesDir, LABEL_PATH);

      OutputStream outStreamInit = new FileOutputStream(outFileInit);
      OutputStream outStreamPredict = new FileOutputStream(outFilePredict);
      OutputStream outStreamLabels = new FileOutputStream(outFileLabels);

      byte[] bufferInit = new byte[1024];
      int readInit;
      while ((readInit = inStreamInit.read(bufferInit)) != -1){
        outStreamInit.write(bufferInit, 0, readInit);
      }
      byte[] bufferPredict = new byte[1024];
      int readPredict;
      while ((readPredict = inStreamPredict.read(bufferPredict)) != -1){
        outStreamPredict.write(bufferPredict, 0, readPredict);
      }
      byte[] bufferLabels = new byte[1024];
      int readLabels;
      while ((readLabels = inStreamLabels.read(bufferLabels)) != -1){
        outStreamLabels.write(bufferLabels, 0, readLabels);
      }

      inStreamInit.close();
      outStreamInit.flush();
      outStreamInit.close();
      Log.d(TAG, "Copied" + INIT_MODEL_PATH + " to " + filesDir);
      String tempPathInit = filesDir + "/" + INIT_MODEL_PATH;

      inStreamPredict.close();
      outStreamPredict.flush();
      outStreamPredict.close();
      Log.d(TAG, "Copied" + PREDICT_MODEL_PATH + " to " + filesDir);
      String tempPathPredict = filesDir + "/" + PREDICT_MODEL_PATH;

      inStreamLabels.close();
      outStreamLabels.flush();
      outStreamLabels.close();
      Log.d(TAG, "Copied" + LABEL_PATH + " to " + filesDir);
      String tempPathLabels = filesDir + "/" + LABEL_PATH;
      LABEL_PATH_LOCAL = tempPathLabels;

      mypredictor = Caffe2.new_(tempPathInit, tempPathPredict, Caffe2.CPUMode, 1);
      if(mypredictor == null){
        Log.e(TAG, "Caffe2.new_ returning null model");
      }
    }catch (Exception e){
      e.printStackTrace();
    }
    labelList = loadLabelList(activity);
    imgData =
            ByteBuffer.allocateDirect(
                    4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
    labelProbArray = new float[1][labelList.size()];
    filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
    Log.d(TAG, "Created a Caffe2 Image Classifier.");
  }

  /* DEMO: Classifies a frame from the preview stream.
  String classifyFrame(Bitmap bitmap) {
    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      return "Uninitialized Classifier.";
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    tflite.run(imgData, labelProbArray);
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // smooth the results
    applyFilter();

    // print the results
    String textToShow = printTopKLabels();
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }*/

  String classifyFrame(Bitmap bitmap) {
    if (mypredictor == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      return "Uninitialized Classifier.";
    }
    // read bitmapped frame into imgData (ByteBuffer)
    convertBitmapToByteBuffer(bitmap);
    // convert ByteBuffer[] into byte[]
    // as gomobile only supports []byte
    imgData.rewind();
    byte[] imgDataBytes = new byte[imgData.remaining()];
    try {

      // DEBUG - NOT PRINTING => meaning there is no corruption of data
      if(imgData.getFloat(2) == 0.0){
        Log.d(TAG,"imgData is null - WHY ?????");
      }

      imgData.get(imgDataBytes, 0, imgDataBytes.length);

      // DEBUG - NOT PRINTING => meaning imgDataBytes were transferred correctly
      if(imgDataBytes.length != 0){
        Log.d(TAG,"imgDataBytes length = " + Float.toString(imgDataBytes.length));
      }

    }catch (Exception e){
      e.printStackTrace();
    }

    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    // DEBUG - COMMENTING BOTH predict() and readPredictionOutput()
    // does not result in an error
    // try uncommenting one of them (first one)
    try {
      Caffe2.predict(mypredictor, imgDataBytes);
    }catch(Exception e){
      e.printStackTrace();
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // smooth the results
    //applyFilter();

    // print the results
    //String textToShow = printTopKLabels();
    String labelOutput = "";
    try {
      Log.d(TAG, "CALLING readPredictedOutput");
      labelOutput = Caffe2.readPredictionOutput(mypredictor, LABEL_PATH_LOCAL);
    }catch(Exception e){
      e.printStackTrace();
    }

    String textToShow = " labelOutput: " + labelOutput;
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  void applyFilter(){
    int num_labels =  labelList.size();

    // Low pass filter `labelProbArray` into the first stage of the filter.
    for(int j=0; j<num_labels; ++j){
      filterLabelProbArray[0][j] += FILTER_FACTOR*(labelProbArray[0][j] -
                                                   filterLabelProbArray[0][j]);
    }
    // Low pass filter each stage into the next.
    for (int i=1; i<FILTER_STAGES; ++i){
      for(int j=0; j<num_labels; ++j){
        filterLabelProbArray[i][j] += FILTER_FACTOR*(
                filterLabelProbArray[i-1][j] -
                filterLabelProbArray[i][j]);

      }
    }

    // Copy the last stage filter output back to `labelProbArray`.
    for(int j=0; j<num_labels; ++j){
      labelProbArray[0][j] = filterLabelProbArray[FILTER_STAGES-1][j];
    }
  }

  /* DEMO: Closes tflite to release resources.
  public void close() {
    tflite.close();
    tflite = null;
  }*/

  public void close() {
    Caffe2.close(mypredictor);
  }

  /* DEMO: Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  /* DEMO: Memory-map the model file in Assets.
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }*/

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel++];
        imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private String printTopKLabels() {
    for (int i = 0; i < labelList.size(); ++i) {
      sortedLabels.add(
          new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }
    String textToShow = "";
    final int size = sortedLabels.size();
    for (int i = 0; i < size; ++i) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      textToShow = String.format("\n%s: %4.2f",label.getKey(),label.getValue()) + textToShow;
    }
    return textToShow;
  }
}
