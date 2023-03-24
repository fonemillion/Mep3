import pickle
import matplotlib.pyplot as plt











#filename = 'wwc.pickle'
#filename = 'CompInproperSolver.pickle'
#filename = 'FPTvsGreedy.pickle'
filename = 'temp1temp2.pickle'
#filename = 'solver3.pickle'
with open(filename, 'rb') as f:
    [time1,time2, ret] = pickle.load(f)


print(set(ret))
#time1 = [time1[i] for i in range(len(ret)) if ret[i] == 0]
#time2 = [time2[i] for i in range(len(ret)) if ret[i] == 0]
print(sum(time1),sum(time2))
print(len(time1))
fig, ax = plt.subplots()
plt.plot(time1, time2, 'o')
plt.plot([0, max(time2)], [0, max(time2)], '-')
plt.show()

plt.loglog(time1, time2, 'o')
plt.loglog([0.001, 0.01, 0.1,1,10,100,1000],[0.001, 0.01, 0.1,1,10,100,1000], '-')
plt.show()