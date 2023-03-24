import pickle
import matplotlib.pyplot as plt
import copy
from DataSetPick import toData2

filename = 'testGNN4.pickle'
with open(filename, 'rb') as f:
    dataPickle = pickle.load(f)


diff = []
corr1 = 0
corr2 = 0
corr3 = 0
noSol1 = 0
noSol2 = 0
noSol3 = 0
diff1 = 0
diff2 = 0
diff3 = 0
delList = []
for i in dataPickle:
    if i[0] < i[3] or i[1] < i[3] or i[2] < i[3]:
        print('found incorr', i)
        delList.append(i)
    if i[3] == 1:
        delList.append(i)
for i in delList:
    dataPickle.remove(i)



for i in dataPickle:
    diff.append([i[0]-i[3],i[1]-i[3],i[2]-i[3]])
    diff1 += i[0]-i[3]
    diff2 += i[1]-i[3]
    diff3 += i[2]-i[3]
    if i[0] == i[3]:
        corr1 += 1
    if i[1] == i[3]:
        corr2 += 1
    if i[2] == i[3]:
        corr3 += 1
    if i[0] == 100:
        noSol1 += 1
    if i[1] == 100:
        noSol2 += 1
    if i[2] == 100:
        noSol3 += 1


print(diff)
print(corr1,corr2,corr3)
print(noSol1,noSol2,noSol3)
print(diff1,diff2,diff3)
print(diff1 - 100*noSol1,diff2 - 100*noSol2,diff3- 100*noSol3)

for i in dataPickle:
    if i[2] != i[3]:
        print(i)