import matplotlib.pyplot as plt
import numpy as np

# create the figure
fig = plt.figure()

# add axes
ax = fig.add_subplot(111,projection='3d')

xx, yy = np.meshgrid(range(10), range(10))
z = (9 - xx - yy) / 2 

# plot the plane
ax.plot_surface(xx, yy, z, alpha=0.5)

plt.show()