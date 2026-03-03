from matplotlib import pyplot as plt

plt.plot(['0.0001', '0.001', '0.01', '0.1'], [
    0.07774406764656305,
    0.07537635997869074,
    0.08314533298835158,
    0.12337480997666717
])
plt.xlabel('Weight decay')
plt.ylabel('Test loss')
plt.show()