import {IMAGE_H, IMAGE_W, NUM_CLASSES} from './data.js';

export function create_CNN(size){
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));


  if (size > 1){
    
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    
    if (size > 2){
      
      model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
      model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    
    }
  }

  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({units: 64, activation: 'relu'}));

  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}