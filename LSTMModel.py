from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Bidirectional, Dense
import os

class LSTMModel(object):
    def __init__(self, inSeqLen, outSeqLen, numFeatures, lstm_units):
        self.inSeqLen = inSeqLen
        self.outSeqLen = outSeqLen
        self.numFeatures = numFeatures

        model = Sequential()

        model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=(inSeqLen, numFeatures)))
        for i in range(1, len(lstm_units) - 1):
            model.add(LSTM(lstm_units[i], return_sequences=True))
        model.add(LSTM(lstm_units[-1]))
        model.add(Dense(outSeqLen))
        model.compile(loss='mean_absolute_percentage_error', optimizer='sgd', metrics=['accuracy'])

        self.model = model
    
    def loadModel(self, modelFile, weightsFile):
        json_file = open(modelFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weightsFile)
        loaded_model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=[self.numNear, self.numFar ])
        # loaded_model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])
        self.model = loaded_model
        # self.inSeqLen = inSeqLen
        # self.outSeqLen = outSeqLen
        # self.numFeatures = numFeatures
        self.inSeqLen = loaded_model.layers[0].get_input_at(0).get_shape().as_list()[1]
        self.outSeqLen = loaded_model.layers[-1].get_output_at(0).get_shape().as_list()[1]
        self.numFeatures = loaded_model.layers[0].get_input_at(0).get_shape().as_list()[2]
        print ("inSeqLen={}, outSeqLen={}, numFeatures={}".format(self.inSeqLen, self.outSeqLen, self.numFeatures))

    def saveModel(self, outputDir, filePrefix):
        outFilename_model = filePrefix + '.json'
        outFilepath = os.path.join(outputDir, outFilename_model)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(outFilepath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        outFilename_weights = filePrefix + '.h5'
        outFilepath = os.path.join(outputDir, outFilename_weights)
        self.model.save_weights(outFilepath)
        print("Saved model to disk")
    
    def train(self, X, y, epochs, batchSize, validationSplit):
        self.model.fit(X, y, validation_split=validationSplit, epochs=epochs, batch_size=batchSize, verbose=2)
    
    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, verbose=2)
        print("Accuracy: ", score[1]*100)