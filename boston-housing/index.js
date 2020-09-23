const tf = require('@tensorflow/tfjs');

const BostonHousingDataset = require('./data');
const normalization = require('./normalization');

// Some hyperparameters for model training.
const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};

function arraysToTensors() {
    tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
    tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
    tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
    tensors.testTarget = tf.tensor2d(bostonData.testTarget);
    // Normalize mean and standard deviation of data.
    let { dataMean, dataStd } =
        normalization.determineMeanAndStddev(tensors.rawTrainFeatures);

    tensors.trainFeatures = normalization.normalizeTensor(
        tensors.rawTrainFeatures, dataMean, dataStd);
    tensors.testFeatures =
        normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);
};

function linearRegressionModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [bostonData.numFeatures], units: 1
    }));
    model.summary();
    return model;
};

function multiLayerPerceptronRegressionModel1Hidden() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [bostonData.numFeatures],
        units: 50,
        activation: 'relu',
        kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dense({units: 1}));
    model.summary();
    return model;
};

function multiLayerPerceptronRegressionModel2Hidden() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [bostonData.numFeatures],
        units: 50,
        activation: 'sigmoid',
        kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dense({
        units: 75, 
        activation: 'sigmoid', 
        kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dense({
        units: 25, 
        activation: 'sigmoid', 
        kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dense({ units: 1 }));

    model.summary();
    return model;
};


function describeKernelElements(kernel) {
    tf.util.assert(
        kernel.length == 12,
        `kernel must be a array of length 12, got ${kernel.length}`);
    const outList = [];
    for (let idx = 0; idx < kernel.length; idx++) {
        outList.push({description: bostonData.featureDescriptions[idx], value: kernel[idx]});
    }
    return outList;
}

async function run(model, modelName) {
    model.compile({ 
        optimizer: tf.train.sgd(LEARNING_RATE), 
        loss: 'meanSquaredError'
    });

    let trainLogs = [];

    console.log('Starting training process...');
    await model.fit(tensors.trainFeatures, tensors.trainTarget, {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS,
        validationSplit: 0.2,
        callbacks: {
        onEpochEnd: async (epoch, logs) => {
            console.log(
                `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`, modelName);
            trainLogs.push(logs);
        }
    }});

    console.log('Running on test data...');
    const result = model.evaluate(
        tensors.testFeatures, tensors.testTarget, {batchSize: BATCH_SIZE});
    const testLoss = result.dataSync()[0];

    const trainLoss = trainLogs[trainLogs.length - 1].loss;
    const valLoss = trainLogs[trainLogs.length - 1].val_loss;
    console.log(
        `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
        `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
        `Test-set loss: ${testLoss.toFixed(4)}`,
        modelName
    );
    
    console.log(`Targets: [${bostonData.testTarget[0]}, ${bostonData.testTarget[1]}]`);
    console.log('predictions:');
    model.predict(
        tf.tensor2d([bostonData.testFeatures[0], bostonData.testFeatures[1]])
    ).print();
};

function computeBaseline() {
    const avgPrice = tensors.trainTarget.mean();
    console.log(`Average price: ${avgPrice.dataSync()}`);
    const baseline = tensors.testTarget.sub(avgPrice).square().mean();
    console.log(`Baseline loss: ${baseline.dataSync()}`);
    const baselineMsg = `Baseline loss (meanSquaredError) is ${
        baseline.dataSync()[0].toFixed(2)}`;
    console.log(baselineMsg);
};

(async () => {
    await bostonData.loadData();
    arraysToTensors();
    computeBaseline();
    await run(multiLayerPerceptronRegressionModel2Hidden(), 'multiLayerPerceptronRegressionModel2Hidden');
})();

