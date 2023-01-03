import matplotlib.pyplot as plt
from pickle import load

with open('accs.pt','rb') as f:
    accs = load(f)

print(accs)

plt.plot(accs)
plt.show()