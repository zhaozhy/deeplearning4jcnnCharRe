package com.test.CharRecognition;

import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public class MyDataSetIterator implements MultiDataSetIterator {
    private  int batchSize=0;
    private  int batchNum=0;
    private  int numExample=0;
    private  MyDataLoader loader;
    private  MultiDataSetPreProcessor preProcessor;

    public  MyDataSetIterator(int batchSize,String dataSetType){
      this (batchSize,null,dataSetType);
    }

    public  MyDataSetIterator(int batchSize, ImageTransform imageTransform,String dataSetType){
        this .batchSize=batchSize;
        loader =new MyDataLoader(imageTransform,dataSetType);
        numExample=loader.totalExamples();
    }


    @Override
    public MultiDataSet next(int num) {
        batchNum +=num;
        MultiDataSet mds=loader.next(num);
        if(preProcessor!=null){
            preProcessor.preProcess(mds);
        }
        return mds;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this .preProcessor =preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return  preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        batchNum=0;
        loader.reset();
    }

    @Override
    public boolean hasNext() {
        if(batchNum< numExample){
            return true;
        }
        else {
            return false;
        }

    }

    @Override
    public MultiDataSet next() {
        return  next(batchSize);
    }
}
