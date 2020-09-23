
module.exports.determineMeanAndStddev = (data) => {
    const dataMean = data.mean(0);
    const diffFromMean = data.sub(dataMean);
    const squaredDiffFromMean = diffFromMean.square();
    const variance = squaredDiffFromMean.mean(0);
    const dataStd = variance.sqrt();
    return { dataMean, dataStd };
}

module.exports.normalizeTensor = (data, dataMean, dataStd) => {
    return data.sub(dataMean).div(dataStd);
}
