# "* ==========================================================",
# "* Description: visual.py",
# "* All rights reserved.",
# "* Date: 2023/12/22 20:38",
# "* ==========================================================",
import matplotlib.pyplot as plt
import numpy as np


def visualise(y_test, y_pred):
    x_line = np.linspace(0, 5, 100)
    y_line = x_line
    plt.plot(x_line, y_line, '-r')
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()
