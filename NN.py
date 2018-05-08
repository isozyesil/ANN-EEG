import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import logistic
from scipy.stats import logistic
import math
import matplotlib.pyplot as plt

"""
Error : Target - Calculated
Output : Using Activation Function
Result : Direct result of Output without ActivationFunction
Input : Each Epoch as array
ActivationFunctions, Functions:  Name of Activation Functions
TargetStartingIndex = Target starting column in an epoch 
TargetEnding Index = Target ending column in an epoch

"""


def CSVCall(Path):  # Calls CSV File and Assign it Input as Numpy Array

    DataSet = pd.read_csv(Path, header=None, delimiter=';')

    Input = np.array(DataSet)

    return Input


def Weight(min, max,
           numberofWeight):  # After you have entered number of nodes and layers for ANN calculation, this method generates  weights,
    # It generates random numbers. Min Max value interval is the random number
    # It returns as array
    t = 0
    Weights = list()

    for Counter in range(0, len(numberofWeight) - 1, 1):
        GenerateWeights = np.random.uniform(min, max, (numberofWeight[Counter] * numberofWeight[Counter + 1]))

        Weights.append(GenerateWeights)

    return Weights


def Bias(min, max, numberofWeight):  # Same logic as weights
    Biases = np.random.uniform(min, max, numberofWeight, )

    return Biases


def Activations(Functions, Output):
    if Functions == 'Tanh':
        return np.tan(Output)

    elif Functions == 'LogSig':
        T = 1 / (1 + np.exp((np.array(Output)) * -1))

        return T
    elif Functions == 'SoftMax':

        T = np.exp(np.array(Output) - np.max(Output))

        T = T / T.sum()

        return T
    elif Functions == 'ReLu':
        T = np.maximum(Output, 0)
        return T


def Activation_Derrivatives(Functions, Output):
    if Functions == 'Tanh':
        T = 1.0 - np.tanh(Output) ** 2
        return T

    elif Functions == 'LogSig':

        T = 1 / (1 + np.exp((np.array(Output)) * -1))

        T = T * (1.0 - T)

        return T

    elif Functions == 'SoftMax':
        Output = np.clip(Output, -500, 500)
        Output = Activations('SoftMax', Output)

        return np.multiply(Output, 1 - Output)

    elif Functions == 'ReLu':
        T = np.greater(Output, 0).astype(int)
        return T


def FeedForwardCalculation(Inputs, Weights, Biases, Layers, ActivationFunctions):
    U = Inputs
    i = 0
    j = 1
    Counter = -1
    CountItem = len(Layers)
    t = 0
    l = 0
    p = -1
    k = 0
    T = list()
    J = list()

    Sum = [0]
    Output = []

    while (Counter != Layers[i]) and (i != CountItem):

        Counter = Counter + 1
        p = p + 1

        Sum = [0]

        if Counter == Layers[i]:

            Output = np.reshape(Output, (p, Layers[i + 1]))  # reshaping it to provide matrix are in the same order

            Output = np.array(Output).sum(axis=0) + np.array(Biases[i])

            Output = np.array(Output)

            U = Output

            T.append(U)

            U = Activations(ActivationFunctions[t], Output)

            J.append(U)

            Result = U

            i = i + 1

            j = j + 1
            Counter = 0
            t = t + 1

            l = 0
            p = 0
            Output = []

            if CountItem == j:
                break

        for Counter2 in range(0, Layers[j], 1):
            Sum = (np.array(U[p]) * (np.array(Weights[t][l])))  # Weight*Input

            Output.append(Sum)

            l = l + 1

    return (Result, T, J)


def ErrorCalculation(Target, Calculated, FirstNode, LastNode, CalculationMethod):
    Error = list()
    ErrorEachNode = list()
    i = 0
    J = 0

    for Counter in range(FirstNode, LastNode + 1, 1):
        if CalculationMethod == 'CrossEntropy':
            A = np.array(Target[Counter]) * np.log10(Calculated[i])

            B = (1 - np.array(Target[Counter])) * (np.log10(1 - np.array(Calculated[i])))

            C = (A + B) * -1

            ErrorEachNode.append(C)

            i = i + 1

        if CalculationMethod == 'MSE':
            K = Target[Counter] - Calculated[i]
            K = (math.pow(K, 2) / 2)
            ErrorEachNode.append(K)

            i = i + 1

    Error = np.sum(ErrorEachNode, dtype=float)

    return Error, ErrorEachNode


def DerrivateOfOutputs(Error, Output, Result, ActivationFunctions):
    """
    print Error
    print Output  # After Activation Function
    print Result  # All Results before activation function
    print ActivationFunctions
    print len(Result)
    """
    ActivationFunctionDerrivates = list()
    for Counter in range(0, len(Result), 1):
        J = Activation_Derrivatives(ActivationFunctions[Counter], Result[Counter])

        ActivationFunctionDerrivates.append(J)

    return ActivationFunctionDerrivates


def DerrivateofErrors(Target, Calculated, FirstNode, LastNode, CalculationMethod):
    i = 0
    DerrivatesofErrors = list()
    for Counter in range(FirstNode, LastNode + 1, 1):
        if CalculationMethod == 'MSE':
            K = Target[Counter] - Calculated[i]

            DerrivatesofErrors.append(K)


        elif CalculationMethod == 'CrossEntropy':
            K = Target[Counter] * (1 / Calculated[i])

            L = 1 - Target[Counter]

            P = 1 / (1 - Calculated[i])

            T = (K + (L * P)) * -1
            K = Target[Counter] * (1 / Calculated[i])

            L = 1 - Target[Counter]

            P = 1 / (1 - Calculated[i])

            T = (K + (L * P)) * -1

            DerrivatesofErrors.append(T)

            i = i + 1

    return DerrivatesofErrors


def HiddenLayerError(ErrorEachNode, Weights, Error, Layers):
    LenLayers = len(Layers)
    WeightLastLayer = list()

    for Counter in range(0, (Layers[LenLayers - 2] * Layers[LenLayers - 1]), 1):
        WeightLastLayer.append(Weights[1][Counter])

    WeightLastLayer = np.reshape(WeightLastLayer, (Layers[LenLayers - 2], Layers[LenLayers - 1]))

    LenWeightLastLayer = len(WeightLastLayer)

    SumWeightLastLayer = np.sum(WeightLastLayer, axis=0)

    BackPropErrorLastLayer = SumWeightLastLayer * ErrorEachNode

    HiddenLayerErrors = WeightLastLayer / np.transpose(BackPropErrorLastLayer)
    return HiddenLayerErrors


def UpdateOutputWeights(DerrivatesofEachError, AFResult, DerrivativesofAF, Error):
    LenLayers = len(Layers)

    DeltaWeights = np.array(DerrivatesofAF[LenLayers - 2]) * np.array(DerrivatesofEachError)

    DeltaWeights = np.array(DeltaWeights).reshape(-1, 1) * AFResult[LenLayers - 3]  # test et

    DeltaWeights = np.ravel(DeltaWeights)

    return DeltaWeights


def UpdateHiddenLayersWeights(Error, ErrorEachNode, DerrivatesofEachError, Output, AllResults, AFResult,
                              DerrivativesofAF,
                              Weights, LearningRate, Layers,
                              lenLastLayer):
    i = 1
    LenDerrivatesofAF = len(DerrivatesofAF)
    LenAFResult = len(AFResult)
    LenWeight = len(Weights)
    LenLayers = len(Layers)

    # print DerrivatesofEachError

    # print Weights[LenWeight - 1]

    DerrivatesofEachError = np.array(DerrivatesofEachError).reshape(-1, 1)
    DerrivatesofAF[LenDerrivatesofAF - 1] = np.array(DerrivatesofAF[LenDerrivatesofAF - 1]).reshape(-1, 1)
    T = list()
    L = list()

    for Counter in range(1, LenWeight - 1, 1):
        K = DerrivatesofEachError * (DerrivatesofAF[LenWeight - 1])

        K = np.array(Weights[LenWeight - Counter]).reshape(Layers[LenLayers - (Counter + 1)],
                                                           Layers[LenLayers - (Counter + 2)]) * np.transpose(K)
        K = K.sum(axis=1)

        L = np.transpose(AFResult[LenAFResult - (Counter + 2)]).reshape(-1, 1) * np.array(
            DerrivatesofAF[LenDerrivatesofAF - (Counter + 1)]).reshape(-1, 1) * K

    return L


def UpdateInputWeight(Error, ErrorEachNode, DerrivatesofEachError, Output, AllResults, AFResult, DerrivatesofAF,
                      Weights, LearningRate, Layers, lenLastLayer, Input, HiddenLayerErrors):
    LenInput = len(Input)

    LenErrorEachNode = len(ErrorEachNode)

    L = list()

    for Counter in range(0, LenInput - LenErrorEachNode, 1):
        K = Input[Counter]
        L.append(K)

    Input = L

    LenDerrivatesofAF = len(DerrivatesofAF)
    LenAFResult = len(AFResult)
    LenWeight = len(Weights)
    LenLayers = len(Layers)

    UpdatedInputWeights = LearningRate * np.sum((HiddenLayerErrors), axis=1) * np.array(DerrivatesofAF[0]) * np.array(
        Input).reshape(-1, 1)

    return UpdatedInputWeights
    """

    
    DerrivatesofEachError = np.array(DerrivatesofEachError).reshape(-1, 1)
    DerrivatesofAF[LenDerrivatesofAF - 1] = np.array(DerrivatesofAF[LenDerrivatesofAF - 1]).reshape(-1, 1)
    T = list()
    K = DerrivatesofEachError * (DerrivatesofAF[LenWeight - 1])

    K = np.array(Weights[LenWeight - 1]).reshape(Layers[1], Layers[2]) * np.transpose(K)

    K = K.sum(axis=1)
    L = np.transpose(Input).reshape(-1, 1) * np.transpose(DerrivatesofAF[0]) * K
    return L
    """

def UpdateAllWeights(UpdatedOutputWeights, UpdatedHiddenWeights, UpdatedInputWeights, Weights, LearningRate):
    DeltaWeights = []

    UpdatedInputWeights = np.ravel(UpdatedInputWeights)

    if len(Weights) > 2:
        UpdatedHiddenWeights = np.ravel(UpdatedHiddenWeights)
        DeltaWeights.append(UpdatedHiddenWeights)

    UpdatedOutputWeights = np.ravel(UpdatedOutputWeights)

    DeltaWeights.append(UpdatedInputWeights)

    DeltaWeights.append(UpdatedOutputWeights)

    DeltaWeights = np.array(DeltaWeights) * LearningRate

    Weights = np.add(Weights, DeltaWeights)

    return Weights


def Normalization(Data, InputStartingIndex, InputEndingIndex):
    """
    J = list()
    for Counter in range(InputStartingIndex,InputEndingIndex+1,1):
        K = Data[Counter]
        J.append(K)

    J = np.array(J)
    Data_normed = (J - J.min(0)) / J.ptp(0)

    for Counter in range(InputEndingIndex+1,len(Data),1):


        Data_normed=np.append(Data_normed,Data[Counter])

    """
    Data_normed = (Data - Data.min(0)) / Data.ptp(0)

    return Data_normed


def ShuffleDataSet(Data, Test, Validation):

    LenTestData = len(Data) * Test / 100
    LenValidationData = len(Data) * Validation / 100

    Data = np.random.permutation(Data)

    return Data, LenTestData, LenValidationData


def ValidationCheck(Input, TestLenData, ValidationLenData, InputStartingIndex, InputEndingIndex, Weights, Biases,
                    Layers, ActivationFunctions, TargetStartingIndex, TargetEndingIndex, CalculationMethod):
    print "Validation Started"

    for Counter in range(len(Input) - TestLenData - ValidationLenData, len(Input) - TestLenData, 1):
        # Input[Counter] = Normalization(Input[Counter], InputStartingIndex, InputEndingIndex)
        (Output, AllResults, AFResult) = FeedForwardCalculation(Input[Counter], Weights, Biases, Layers,
                                                                ActivationFunctions)  # 1

        # for Counter2 in range(TargetStartingIndex, TargetEndingIndex + 1, 1):
        #    print Input[Counter][Counter2]

        (Error, ErrorEachNode) = ErrorCalculation(Input[Counter], Output, TargetStartingIndex, TargetEndingIndex,
                                                  CalculationMethod)
        print Error


def TestCheck(Input, TestLenData, ValidationLenData, InputStartingIndex, InputEndingIndex, Weights, Biases, Layers,
              ActivationFunctions, TargetStartingIndex, TargetEndingIndex, CalculationMethod):
    L = list()
    Counter3 = 0
    print "Test Started"

    for Counter in range(len(Input) - TestLenData, len(Input), 1):
        Input[Counter] = Normalization(Input[Counter], InputStartingIndex, InputEndingIndex)
        (Output, AllResults, AFResult) = FeedForwardCalculation(Input[Counter], Weights, Biases, Layers,
                                                                ActivationFunctions)  # 1

        Output== Normalization(Output, InputStartingIndex, InputEndingIndex)

        print Output



        for Counter2 in range(TargetStartingIndex, TargetEndingIndex + 1, 1):
            print Input[Counter][Counter2]
            L.append(Input[Counter][Counter2])




        print np.argmax(L)
        print np.argmax(Output)


        if (np.argmax(L)==np.argmax(Output)):
            Counter3=(Counter3)+1
            M= (float(TestLenData))/float(100*Counter3)
            print M


        L=[]





# (Error,ErrorEachNode)=ErrorCalculation(Input[Counter], Output, TargetStartingIndex, TargetEndingIndex,
#                 CalculationMethod)


def P300Calculator(Input, SampleRate):
    LenInput = len(Input)
    K = list()
    T = list()

    for Counter in range(1, LenInput + 1, 1):
        K.append(Input[Counter - 1])

        if len(K) % SampleRate == 0:
            J = np.array(K).sum(axis=0)

            J = J / SampleRate
            K = []
            T.append(J)

            J = []

    return T

def SuccessCheck(Input,TestLenData,ValidationLenData,Output,Error,TargetStartingIndex,TargetEndingIndex,SuccessCounter,Counter,Counter2):
    L = list()

    LenInput=len(Input)


    for Counter4 in range(TargetStartingIndex, TargetEndingIndex + 1, 1):
       # print Input[Counter][Counter2]
        L.append(Input[Counter][Counter4])

    #print np.argmax(L), np.argmax(Output)


    if (np.argmax(L) == np.argmax(Output)):
        SuccessCounter = (SuccessCounter) + 1
        M =  float(100.0 * float(SuccessCounter))/((LenInput*(1+Counter2)))
        print M

    L = []

    return SuccessCounter



FileLocation = 'venv/DataSet.csv'
Input = CSVCall(FileLocation)

Biases = Bias(-0.1, +0.1, 15)
Layers = [16, 10, 5]
Weights = Weight(-0.1, 0.1, Layers)
SuccessCounter=0
AllResults = list()
AFResult = list()
Output = list()
ActivationFunctions = ['LogSig', 'LogSig']
LearningRate = 0.1
# Weights = [[0.15, 0.25, 0.20, 0.30], [0.40, 0.50, 0.45, 0.55]]
# Biases = [[0.35, 0.35], [0.6, 0.6]]
# Input=[]
# Input = [0.05, 0.1, 0.01, 0.99]
TargetStartingIndex = 16
TargetEndingIndex = 20
InputStartingIndex = 0
InputEndingIndex = 15
lenLastLayer = Layers[len(Layers) - 1]
CalculationMethod = 'MSE'
DeltaWeights = list()
P300SamplingRate =30

"""
plt.ion()  ## Note this correction
fig = plt.figure()
plt.axis([0, 1000, 0, 1])

i = 0
x = list()
y = list()
"""

Input = P300Calculator(Input, P300SamplingRate)

Input, TestLenData, ValidationLenData = ShuffleDataSet(Input, 20, 0)


for Counter2 in range(0, 4, 1):
    for Counter in range(0, len(Input) - TestLenData - ValidationLenData, 1):
        Input[Counter] = Normalization(Input[Counter], InputStartingIndex, InputEndingIndex)
        (Output, AllResults, AFResult) = FeedForwardCalculation(Input[Counter], Weights, Biases, Layers,
                                                                ActivationFunctions)



        (Error, ErrorEachNode) = ErrorCalculation(Input[Counter], Output, TargetStartingIndex, TargetEndingIndex,
                                                  CalculationMethod)

        DerrivatesofAF = DerrivateOfOutputs(Error, Output, AllResults, ActivationFunctions)

        DerrivatesofEachError = DerrivateofErrors(Input[Counter], Output, TargetStartingIndex, TargetEndingIndex,
                                                  CalculationMethod)

        HiddenLayerErrors = HiddenLayerError(ErrorEachNode, Weights, Error, Layers)

        UpdatedOutputWeights = UpdateOutputWeights(DerrivatesofEachError, AFResult, DerrivatesofAF, Error)
        # ravel function will be used for subs.

        UpdatedHiddenWeights = UpdateHiddenLayersWeights(Error, ErrorEachNode, DerrivatesofEachError, Output,
                                                         AllResults,
                                                         AFResult, DerrivatesofAF, Weights,
                                                         LearningRate, Layers, lenLastLayer)

        UpdatedInputWeights = UpdateInputWeight(Error, ErrorEachNode, DerrivatesofEachError, Output, AllResults,
                                                AFResult,
                                                DerrivatesofAF, Weights,
                                                LearningRate, Layers, lenLastLayer, Input[Counter], HiddenLayerErrors)

        Weights = UpdateAllWeights(UpdatedOutputWeights, UpdatedHiddenWeights, UpdatedInputWeights, Weights,
                                   LearningRate)

        SuccessCounter= SuccessCheck(Input, TestLenData, ValidationLenData, Output, Error, TargetStartingIndex, TargetEndingIndex,SuccessCounter,Counter,Counter2)

        #print Error, Counter, Counter2, Input[Counter][16], Input[Counter][17], Input[Counter][18], Input[Counter][19], \
        #Input[Counter][20]



ValidationCheck(Input, TestLenData, ValidationLenData, InputStartingIndex, InputEndingIndex, Weights, Biases, Layers,
                ActivationFunctions, TargetStartingIndex, TargetEndingIndex, CalculationMethod)

TestCheck(Input, TestLenData, ValidationLenData, InputStartingIndex, InputEndingIndex, Weights, Biases, Layers,
          ActivationFunctions, TargetStartingIndex, TargetEndingIndex, CalculationMethod)
