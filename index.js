import {IMAGE_H, IMAGE_W, NUM_CLASSES, NUM_TEST_ELEMENTS, MnistData} from './data.js';
import {create_CNN} from './basicCNN.js';

/* LOAD DATA */
var data;
async function load_MNIST(){
    data = new MnistData();
    await data.load();
    document.getElementById('data_sel').innerHTML += '<br /><br />MNIST LOADED.';
}
window.load_MNIST = load_MNIST;


/* MODEL: GLOBAL SCOPE */
var model;

/* MODELS */
function createDenseModel() {
  const ml = tf.sequential();
  ml.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  ml.add(tf.layers.dense({units: 42, activation: 'relu'}));
  ml.add(tf.layers.dense({units: NUM_CLASSES, activation: 'softmax'}));
  return ml;
}

/* MODEL SELECTION */
function load_dnn(){
    let start = Date.now()
    model = createDenseModel();
    let end = Date.now();
    let loadtime = (end - start).toString();
    document.getElementById('model_sel').innerHTML += '<br /><br />DNN LOADED. LOAD TIME: ' + loadtime;

}
window.load_dnn = load_dnn;

function load_cnn(){
    // Replace with adequate CNN size.
    model = create_CNN(1);
    document.getElementById('model_sel').innerHTML += '<br /><br />CNN LOADED';

}
window.load_cnn = load_cnn;

function load_med_cnn(){
      // Replace with adequate CNN size.
    model = create_CNN(2);
    document.getElementById('model_sel').innerHTML += '<br /><br />MED CNN LOADED';
}
window.load_med_cnn = load_med_cnn;

function load_big_cnn(){
      // Replace with adequate CNN size.
    model = create_CNN(3);
    document.getElementById('model_sel').innerHTML += '<br /><br />BIG CNN LOADED';
}
window.load_big_cnn = load_big_cnn;


/* TRAINING MODELS */
var history;
var times = [];

function write_logs(epoch, logs){
  document.getElementById('outputjs').innerHTML += "<br />" + epoch.toString() + "<br />";
  for (var key in logs) {
    if (logs.hasOwnProperty(key)) {
        document.getElementById('outputjs').innerHTML += key + ':' + logs[key] + '<br />';  
      }
  }
  document.getElementById('outputjs').innerHTML += "<br />";
}


function train(){
  
  document.getElementById('outputjs').innerHTML = "<br/ >TRAINING STARTED<br />";
  
  let optimizer = 'adam';
  
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  
  const batchSize = parseInt(document.getElementById('batch_size').value);
  const trainEpochs = parseInt(document.getElementById('epochs').value);
  const validationSplit = 0.15;
  
  let trainBatchCount = 0;
  const trainData = data.getTrainData();
  
  const totalNumBatches =
      Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
      trainEpochs;
  
  
  let valAcc;
  model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        times.push(Date.now());
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        write_logs(epoch, logs);
        await tf.nextFrame();
      }
    }
  }).then(function(){
    let batchtimes = [];
    for(var i=1; i<times.length-1; i++){
      batchtimes.push(times[i+1]-times[i]);
    }
    let sum = batchtimes.reduce((previous, current) => current += previous);
    let avg_time = sum / batchtimes.length;
    let avg_throughput = batchSize / avg_time;
    document.getElementById('outputjs').innerHTML += "<br /> AVERAGE TRAINING THROUGHPUT: " + avg_throughput.toString();
    
  });
  
}
window.train = train;


/* TESTING MODELS */
function test(){
  console.log("TESTING");
  const testData = data.getTestData();

  var start_time = Date.now();
  const testResult = model.evaluate(testData.xs, testData.labels);
  var end_time = Date.now();
  
  // * 10 because of 1-hot
  let throughput = NUM_TEST_ELEMENTS / (end_time - start_time);
  let time = (end_time - start_time).toString();
  
  document.getElementById('outputtestjs').innerHTML = "<br />THROUGHPUT:TIME -> " + throughput.toString() + ":" + TIME;
}
window.test = test;