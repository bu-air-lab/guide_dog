import numpy as np
import matplotlib.pyplot as plt
 
plt.matplotlib.rcParams['pdf.fonttype'] = 42
plt.matplotlib.rcParams['ps.fonttype'] = 42
plt.matplotlib.rcParams.update({'font.size': 13})

def read_file(filename):

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

    return accuracy, accuracy_std, false_positives, false_positives_std


full_state = 'force_detector_accuracy.txt'
base_vel_only = 'force_detector_only_vel_accuracy.txt'

state_accuracy, state_accuracy_std, state_fp, state_fp_std = read_file(full_state)
vel_only_accuracy, vel_only_accuracy_std, vel_only_fp, vel_only_fp_std = read_file(base_vel_only)

#fig, ax = plt.subplots()
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Force Detection Accuracy and False Positive Rate')
x_axis = [25, 30, 35, 40, 45, 50, 55, 60]

state_std_below = []
state_std_above = []

for t, a in enumerate(state_accuracy):
    state_std_below.append(a - state_accuracy_std[t])
    state_std_above.append(a + state_accuracy_std[t])

ax1.plot(x_axis, state_accuracy, label='Full State')
ax1.fill_between(x_axis, state_std_below, state_std_above, alpha=.1)

state_std_below = []
state_std_above = []
for t, f in enumerate(state_fp):
    state_std_below.append(f - state_fp_std[t])
    state_std_above.append(f + state_fp_std[t])

ax2.plot(x_axis, state_fp, label='Full State')
ax2.fill_between(x_axis, state_std_below, state_std_above, alpha=.1)

vel_only_std_below = []
vel_only_std_above = []

for t, a in enumerate(vel_only_accuracy):
    vel_only_std_below.append(a - vel_only_accuracy_std[t])
    vel_only_std_above.append(a + vel_only_accuracy_std[t])

ax1.plot(x_axis, vel_only_accuracy, label='Only Vel')
ax1.fill_between(x_axis, vel_only_std_below, vel_only_std_above, alpha=.1)

vel_only_std_below = []
vel_only_std_above = []

for t, f in enumerate(vel_only_fp):
    vel_only_std_below.append(f - vel_only_fp_std[t])
    vel_only_std_above.append(f + vel_only_fp_std[t])

ax2.plot(x_axis, vel_only_fp, label='Only Vel')
ax2.fill_between(x_axis, vel_only_std_below, vel_only_std_above, alpha=.1)

ax1.legend(bbox_to_anchor=(0.66, 0.87))
ax2.legend()
#fig.legend(bbox_to_anchor=(1.01, 0.75))

ax1.set_ylabel('Accuracy')
ax2.set_ylabel('False Positive Rate')

ax1.set_yticks([-0.2, 0.0, 0.5, 1.0]) 
ax2.set_yticks([-0.2, 0.0, 0.5, 1.0]) 

plt.xlabel('Force applied (in Newtons)')
#plt.ylabel('Accuracy and False Positive Rate')
#plt.title('Force Detection Accuracy')
plt.savefig("plot_force_detector_accuracy.pdf")