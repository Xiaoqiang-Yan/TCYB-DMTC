# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt

str_io = "NMI against ground truth label:"
nmi = re.compile('(?<=NMI against ground truth label:)\s*\d*\.\d*')
acc = re.compile('(?<=ACC against ground truth label:)\s*\d*\.\d*')

with open("/home/sky/code/python2/DCCM/venv/checkpoint/task02/test.txt", "r") as f:
    data = f.read()
list_nmi = []
list_nmi = list(map(float, nmi.findall(data)))
list_acc = list(map(float, acc.findall(data)))
print(list_nmi)
print(list_acc)

nmi_list = np.array(list_nmi)
acc_list = np.array(list_acc)

plt.plot(list_nmi, color = 'red')
plt.plot(list_acc, color = 'blue')
plt.show()