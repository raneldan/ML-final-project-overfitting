import csv
import matplotlib.pyplot as plt

filename = './data.txt'

with open(filename, 'r', encoding="utf8") as csvin:
    csvin = csv.reader(csvin, delimiter=',')
    itercsvin = iter(csvin)
    i = 0
    epoches = []
    train = []
    test = []
    for row in itercsvin:
        if i % 3 == 0:
            epoches.append(float(row[1]))
        elif i % 3 == 1:
            train.append(float(row[1]))
        elif i % 3 == 2:
            test.append(float(row[1]))
        i += 1

plt.figure()
plt.plot(epoches, train, 'b')
plt.plot(epoches, test, 'r')
plt.xlabel("epoches")
plt.ylabel("accuercy")
plt.title("overfitting")
plt.show()