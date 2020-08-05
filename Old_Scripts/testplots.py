import numpy as np
import seaborn as sns
from numpy import log10
import matplotlib.pyplot as plt
import numdifftools as nd
import os


# Plot spectrum on log scale
log_normalized_evals = np.logspace(-3, 3, num=30)
log_normalized_evals = [el * np.random.uniform(0., 1.) for el in log_normalized_evals]
log_normalized_evals = log10(np.divide(log_normalized_evals, float(max(log_normalized_evals))))
spec_plot = sns.distplot(log_normalized_evals, kde=False, rug=True)
spec_plot.set(xlabel=r'log($\lambda / \lambda_{max}$)', ylabel='bin count')
minX, maxX = plt.xlim()
minY, maxY = plt.ylim()
yval = (maxY+minY)/2. * 0.6
spec_plot.annotate('Sloppiness', xy=(minX, yval), xytext=((maxX+minX)/2.*0.8, yval+0.1),  # draws an arrow from one set of coordinates to the other
                   arrowprops=dict(facecolor='black', width=3),  # sets style of arrow and colour
                   annotation_clip=False)  # This enables the arrow to be outside of the plot
spec_plot.annotate('', xy=((maxX+minX)/2.*0.8+1.1, yval), xytext=(maxX, yval),  # draws an arrow from one set of coordinates to the other
                   arrowprops=dict(facecolor='black', width=3),  # sets style of arrow and colour
                   annotation_clip=False)  # This enables the arrow to be outside of the plot
plt.show()

def test_function(x):
    return 2 * np.sin(x[0]) + 3 * np.cos(x[1])

H = nd.Hessian(test_function)([1.3, 0.3])
print(H)

print(os.path.join(os.getcwd(), 'results', 'Sloppiness'))