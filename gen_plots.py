import numpy as np
import matplotlib.pyplot as plt
 
plt.matplotlib.rcParams['pdf.fonttype'] = 42
plt.matplotlib.rcParams['ps.fonttype'] = 42
plt.matplotlib.rcParams.update({'font.size': 15})


filename = 'force_detector_accuracy.txt'

#Read file containing signals
with open(filename) as f:
    lines = [line.rstrip() for line in f]

accuracy = []
accuracy_std = []
false_positives = []
false_positives_std = []

for line in lines:    

    #Convert to list of floats
    line = line.split()
    accuracy.append(float(line[0]))
    accuracy_std.append(float(line[1]))
    false_positives.append(float(line[2]))
    false_positives_std.append(float(line[3]))


fig, ax = plt.subplots()
x_axis = [25, 30, 35, 40, 45, 50, 55, 60]

std_below = []
std_above = []
for t, a in enumerate(accuracy):
    std_below.append(a - accuracy_std[t])
    std_above.append(a + accuracy_std[t])

ax.plot(x_axis, accuracy, label='Accuracy')
ax.fill_between(x_axis, std_below, std_above, alpha=.1)

std_below = []
std_above = []
for t, f in enumerate(false_positives):
    std_below.append(f - false_positives_std[t])
    std_above.append(f + false_positives_std[t])

ax.plot(x_axis, false_positives, label='False Positive Rate')
ax.fill_between(x_axis, std_below, std_above, alpha=.1)

ax.legend()

plt.xlabel('Force applied (in Newtons)')
plt.ylabel('Accuracy and False Positive Rate')
plt.title('Force Detection Accuracy')
plt.savefig("plot_force_detector_accuracy.pdf", bbox_inches='tight')