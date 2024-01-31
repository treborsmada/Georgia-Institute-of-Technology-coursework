import Testing

print "Question 5:"

print "CarData:"
iteration = 0
accuracies = []
while iteration < 5:
    accuracies.append(Testing.testCarData()[1])
    iteration += 1
print "Average:", Testing.average(accuracies)
print "Standard Deviation:", Testing.stDeviation(accuracies)
print "Maximum:", max(accuracies)

print "PenData:"
iteration = 0
accuracies = []
while iteration < 5:
    accuracies.append(Testing.testPenData()[1])
    iteration += 1
print "Average:", Testing.average(accuracies)
print "Standard Deviation:", Testing.stDeviation(accuracies)
print "Maximum:", max(accuracies)

