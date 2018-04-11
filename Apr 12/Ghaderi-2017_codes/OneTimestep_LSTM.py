import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import keras as keras
np.random.seed(1234)
from keras import backend as K


class multiLSTM:
    def __init__(self):
        self.inputHorizon = 24          # number of time steps as input
        self.inOutVecDim = 57           # number of stations
        self.lstmModels = [None for _ in range(6)]
        self.xTest, self.yTest = None, None
        file_dataset = 'MS_winds.dat'
        with open(file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            winds = []
            for line in data:
                winds.append(line)
        self.winds = (np.array(winds)).astype(float)    # all data
        self.winds = self.winds[:, :self.inOutVecDim]   # ensure that only 57 stations data is included
        self.means_stds = [0, 0]
        self.winds, self.means_stds = self.normalize_winds_0_1(self.winds)
        self.validation_split = 0.05                    # 5% data is used to validate
        self.batchSize = 3                              # size of minibatch
        activation = ['sigmoid', 'tanh', 'relu', 'linear']
        self.activation = activation[2]                 # choose activation type
        realRun = 1                                     # flag indicating whether it is training or not (just debugging)
        # percentage of data used for training (saving time for debugging)
        #          model number :           1   2   3   4   5   6
        self.epochs, self.trainDataRate = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[1, 1, 1, 1, 1, 1], 0.005]

    def normalize_winds_0_1(self, winds):
        """normalize based on each station data"""
        windMax = winds.max()
        windMin = winds.min()
        normal_winds = (winds - windMin) / windMax
        mins_maxs = [windMin, windMax]
        return np.array(normal_winds), mins_maxs

    def denormalize(self, vec):
        res = vec * self.means_stds[1] + self.means_stds[0]
        return res

    def loadData_1(self):
        """
        for lstm1, which is 1-hour prediction model
        :return: X_train, y_train
        """
        result = []
        for index in range(len(self.winds) - self.inputHorizon):
            result.append(self.winds[index:index + self.inputHorizon])
        result = np.array(result)  

        trainRow = int(6000 * self.trainDataRate)
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        self.xTest = result[6000:6361, :]
        self.yTest = self.winds[6000 + self.inputHorizon:6361 + self.inputHorizon]
        self.predicted = np.zeros_like(self.yTest)
        return [X_train, y_train]
  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, 57*2, 57, 32, out_nodes]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=False))
        model.add(Dense(output_dim=layers[4]))
        model.add(Activation(self.activation))
    
        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM(self, lstmModelNum):
        if lstmModelNum == 1:
            return self.buildModelLSTM_1()

    def trainLSTM(self, xTrain, yTrain, lstmModelNum):
        """train first LSTM with inputHorizon number of real input values"""
        lstmModel = self.buildModelLSTM(lstmModelNum)
        lstmModel.fit(xTrain, yTrain,
                      batch_size=self.batchSize,
                      nb_epoch=self.epochs[lstmModelNum-1],
                      validation_split=self.validation_split)
        return lstmModel

    def test(self):
        """calculate the predicted values (self.predicted)"""
        for ind in range(len(self.xTest)):
            testInputRaw = self.xTest[ind]
            testInputShape = testInputRaw.shape         # testInputRaw.shape = (12, 57)
            testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            self.predicted[ind] = self.lstmModels[0].predict(testInput)

    def errorMeasures(self, denormalYTest, denormalYPredicted):
        mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
        rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
        nrsme_maxMin = 100 * rmse / (denormalYTest.max() - denormalYTest.min())
        nrsme_mean = 100 * rmse / (denormalYTest.mean())

        return mae, rmse, nrsme_maxMin, nrsme_mean

    def drawGraphStation(self, station, visualise=1, ax=None):
        """draw graph of predicted vs real values"""
        yTest = self.yTest[:, station]
        denormalYTest = self.denormalize(yTest)
        denormalPredicted = self.denormalize(self.predicted[:, station])

        mae, rmse, nrmse_maxMin, nrmse_mean = self.errorMeasures(denormalYTest, denormalPredicted)
        print('station %2d: MAE = %.5f\tRMSE = %.5f\tnrmse_maxMin = %.5f\tnrmse_mean = %.5f'
              % (station+1, mae, rmse, nrmse_maxMin, nrmse_mean))

        if visualise:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(denormalYTest, label='Real')
            ax.plot(denormalPredicted, label='Predicted', color='red')
            ax.set_xticklabels([0, 100, 200, 300, 400], rotation=40)

        return mae, rmse, nrmse_maxMin, nrmse_mean

    def drawGraphAllStations(self):
        """draw graph of predicted vs real values of station no. 1 ~ 16"""
        rows, cols = 4, 4
        maeRmse = np.zeros((rows*cols, 4))

        fig, ax_array = plt.subplots(rows, cols, sharex='all', sharey='all')
        staInd = 0
        for ax in np.ravel(ax_array):       # np.ravel() flatten a np array
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300, 400])
        print(maeRmse.mean(axis=0))

        filename = 'finalEpoch'
        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.show()

    def run(self):
        #  training
        xTrain, yTrain = self.loadData_1()
        print(' Training LSTM 1 ...')
        self.lstmModels[0] = self.trainLSTM(xTrain, yTrain, 1)

        # testing
        print(' TESTING  ...')
        self.test()

        self.drawGraphAllStations()


DeepForecast = multiLSTM()
DeepForecast.run()
