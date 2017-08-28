from ccfeplot import CcfePlot
import matplotlib.pyplot as plt
import numpy as np

cfig = CcfePlot(2,2)

x = np.linspace(-1,4, 100)
y = np.sin(x)

cfig.plot(x,y)
cfig.plot(x,2*y)
plt.show()
print('CCFE Plot demo complete')