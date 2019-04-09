package com.test.CharRecognition;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.api.storage.StatsStorage;

import java.io.File;

public class NumberRecognition {
    private  static  long seed=123;
    private  static  int epochs=50;
    private  static  int batchSize=15;
    private  static  String  rootPath=System.getProperty("user.dir");

    private static  String modelDirPath= rootPath.substring(0,rootPath.lastIndexOf(File.separatorChar))+File.separatorChar+"out"+File.separatorChar+"models";
    private  static  String  validatemodelPath=modelDirPath+File.separatorChar+"validateCodeCheckModel.json";

    public static void main(String[] args) throws Exception {

        long startTime=System.currentTimeMillis();
        System.out.println(startTime);
        File modelDir=new File(modelDirPath);
        boolean hasDir=modelDir.exists() || modelDir.mkdirs();
        System.out.println(validatemodelPath);
        ComputationGraph model=createModel();
        UIServer uiServer=UIServer.getInstance();
        org.deeplearning4j.api.storage.StatsStorage stateStorage=new InMemoryStatsStorage();
        uiServer.attach(stateStorage);
        model.setListeners(new ScoreIterationListener(10),new StatsListener(stateStorage));

        MyDataSetIterator trainIterator=new MyDataSetIterator(batchSize,"train");
        MyDataSetIterator testIterator=new MyDataSetIterator(batchSize,"test");
        MyDataSetIterator validateIterator=new MyDataSetIterator(batchSize,"validate");

        for(int i=0;i<epochs;i++){

            System.out.println("Epoxh============"+i);
            model.fit(trainIterator);
        }

        ModelSerializer.writeModel(model,validatemodelPath,true);
        long endTime=System.currentTimeMillis();
        System.out.println("========run time============"+(endTime-startTime));

        System.out.println("==========eval model======test==========");
        modelPredict(model,testIterator);

        System.out.println("==========eval model ========validate=========");
        modelPredict(model,validateIterator);

    }

    private static ComputationGraph createModel() {
        ComputationGraphConfiguration configuration=new NeuralNetConfiguration.Builder()
            .seed(seed)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .l2(1e-3)
            .updater(new Adam(1e-3))
            .weightInit(WeightInit.XAVIER_UNIFORM)
            .graphBuilder()
            .addInputs("trainFeatures")
            .setInputTypes(InputType.convolutional(60,160,1))
            .setOutputs("out1","out2","out3","out4","out5","out6")
            .addLayer("cnn1",new ConvolutionLayer.Builder(new int [] {5,5},new int []{1,1},new int[]{0,0}).nIn(1).nOut(48).activation(Activation.RELU).build(),"trainFeatures")
            .addLayer("maxpool1",new SubsamplingLayer.Builder(PoolingType.MAX,new int []{2,2},new int []{2,2},new int []{0,0}).build(),"cnn1")
            .addLayer("cnn2",new ConvolutionLayer.Builder(new int[] {5,5},new int[]{1,1},new int[]{0,0}).nOut(64).activation(Activation.RELU).build(),"maxpool1")
            .addLayer("maxpool2",new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,new int []{2,1},new int []{2,1},new int []{0,0}).build(),"cnn2")
            .addLayer("cnn3",new ConvolutionLayer.Builder(new int[] {3,3},new int[]{1,1},new int[]{0,0}).nOut(128).activation(Activation.RELU).build(),"maxpool2")
            .addLayer("maxpool3",new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,new int []{2,2},new int []{2,2},new int []{0,0}).build(),"cnn3")
            .addLayer("cnn4",new ConvolutionLayer.Builder(new int[] {4,4},new int[]{1,1},new int[]{0,0}).nOut(256).activation(Activation.RELU).build(),"maxpool3")
            .addLayer("maxpool4",new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,new int []{2,2},new int []{2,2},new int []{0,0}).build(),"cnn4")
            .addLayer("ffn0",new DenseLayer.Builder().nOut(3072).build(),"maxpool4")
            .addLayer("ffn1",new DenseLayer.Builder().nOut(3072).build(),"ffn0")
            .addLayer("out1",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build(),"ffn1")
            .addLayer("out2",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build(),"ffn1")
            .addLayer("out3",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build(),"ffn1")
            .addLayer("out4",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build(),"ffn1")
            .addLayer("out5",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build(),"ffn1")
            .addLayer("out6",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build(),"ffn1")
            .build();

        ComputationGraph model=new ComputationGraph(configuration);
        model.init();
        return  model;
    }
    public  static  void  modelPredict(ComputationGraph model, MultiDataSetIterator iterator){
        int sumCount=0;
        int correctCount=0;

        while (iterator.hasNext()){
            MultiDataSet mds=iterator.next();
            INDArray[] output=model.output(mds.getFeatures());
            INDArray[] labels=mds.getLabels();
            int dataNum=batchSize>output[0].rows()?output[0].rows():batchSize;
             for(int dataIndex=0;dataIndex<dataNum;dataIndex++){
                  String reLabel="";
                  String peLabel="";
                  INDArray preOutput=null;
                  INDArray realLabel=null;
                  for (int digit=0;digit<6;digit++){
                      preOutput=output[digit].getRow(dataIndex);
                      peLabel+= Nd4j.argMax(preOutput,1).getInt(0);
                      realLabel=labels[digit].getRow(dataIndex);
                      reLabel+= Nd4j.argMax(realLabel,1).getInt(0);
                  }
                  if(peLabel.equals(reLabel)){
                      correctCount++;
                  }
                  sumCount++;
                 System.out.println("real image "+reLabel+"  prediction "+peLabel+" status "+ peLabel.equals(reLabel));
             }
        }
        iterator.reset();
        System.out.println("validate result : sum count =" + sumCount + " correct count=" + correctCount );
    }
}
