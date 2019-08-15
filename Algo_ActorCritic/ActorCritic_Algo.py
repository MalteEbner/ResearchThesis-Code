from keras.models import Model
from keras.layers import Input, Dense






class Policy:
    def __init__(self,actionSpace,stateSpace=0):
        if stateSpace==0:
            self.inputShape = (1,)
        self.actionSpace = actionSpace


    def defineModel(self):
        categoricalOutputs = self.actionSpace.VariantNumbers()
        if self.actionSpace.withScheduleCompression:
            noRealOutputs = self.actionSpace.noActivities
        else:
            noRealOutputs = 0


        '''Model definition of neural network'''
        #input layer
        inputs = Input(shape=self.inputShape)
        #intermediate layers
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)

        #output layers with losses
        outputs = []
        losses = []
        for noVariants in categoricalOutputs:
            variantLayer = Dense(noVariants,activation='softmax')(x)
            outputs.append(variantLayer)
            losses.append('categorical_crossentropy')
        for i in range(noRealOutputs):
            scheduleCompressionLayer = Dense(1)(x)
            outputs.append(scheduleCompressionLayer)
            losses.append('mean_squared_error')

        #define model
        model = Model(inputs=inputs,outputs=outputs)
        model.compile(optimizer='sgd',
                      loss=losses,
                      metrics=['accuracy'])

        self.model = model


    def updateModel(self,output,updateWeight,input=0):
        self.model.fit(input,output,sample_weights=updateWeight)

    def getNextAction(self,input=0):
        output = self.model(input)








