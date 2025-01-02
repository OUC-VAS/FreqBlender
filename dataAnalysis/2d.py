import numpy as np
import matplotlib.pyplot as plt

dataset = 'Face2Face'
average_psd = np.load("")

for i in range(400):
    for j in range(400):
        if average_psd[i][j] > 0:
            average_psd[i][j] = 10 * np.log10(average_psd[i][j])
        else:
            average_psd[i][j] = - 10 * np.log10(-average_psd[i][j])

Y = average_psd[:, 0]

plt.figure()
X = np.arange(0, 400)

plt.plot(X, Y, color="b",linestyle="-")
plt.grid(linestyle="--", alpha=0.5)
# Add a color bar which maps values to colors.
plt.show()
