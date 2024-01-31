import NeuralNet
import Testing
print "Question 6:"
print "PenData"
hiddenLayer = 0
while hiddenLayer <= 40:
    print "HiddenLayer: ", hiddenLayer
    iteration = 0
    accuracies = []
    while iteration < 5:
        accuracies.append(NeuralNet.buildNeuralNet(Testing.penData, maxItr= 200, hiddenLayerList=[hiddenLayer])[1])
        iteration += 1
    print "Average:", Testing.average(accuracies)
    print "Standard Deviation:", Testing.stDeviation(accuracies)
    print "Maximum:", max(accuracies)
    hiddenLayer += 5

print "CarData"
hiddenLayer = 0
while hiddenLayer <= 40:
    print "HiddenLayer: ", hiddenLayer
    iteration = 0
    accuracies = []
    while iteration < 5:
        accuracies.append(NeuralNet.buildNeuralNet(Testing.carData, maxItr= 200, hiddenLayerList=[hiddenLayer])[1])
        iteration += 1
    print "Average:", Testing.average(accuracies)
    print "Standard Deviation:", Testing.stDeviation(accuracies)
    print "Maximum:", max(accuracies)
    hiddenLayer += 5