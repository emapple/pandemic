import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from matplotlib import animation
from matplotlib import collections as clt
from ball import *


class UpdatablePatchCollection(clt.PatchCollection):
    """Updatable patch collection

    Borrowed from 
    https://stackoverflow.com/questions/48794016/animate-a-collection-of-patches-in-matplotlib
    """

    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        clt.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class sim:
    """Simulation object"""

    def __init__(self, n_ball, ndim, **params):

        self.ndim = ndim
        self.n_ball = n_ball

        self.balls = ballCollection(self.n_ball, self.ndim, **params)

        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(self.balls.corners[0])
        if self.ndim > 1:
            self.ax.set_ylim(self.balls.corners[1])
            xspan = float(np.diff(self.balls.corners[0]))
            yspan = float(np.diff(self.balls.corners[1]))

            scaler = 5. / xspan
            xspan *= scaler
            yspan *= scaler
            self.fig.set_size_inches(xspan, yspan)
            self.ax.set_aspect('equal')
        else:
            self.ax.set_ylim([-1, 1])
            self.fig.set_size_inches(5, 1)

        x, y = self.balls.get_xy()
        self.patches = [plt.Circle((ix, iy), self.balls.size)
                        for ix, iy in zip(x, y)]
        self.collection = UpdatablePatchCollection(
            self.patches, edgecolor='none', facecolor='none')
        self.collection.set_array(np.zeros(len(self.balls.balls)))
        self.ax.add_collection(self.collection)
        self.started = False
        self.text = self.ax.text(0.5, 0.5, 'Click anywhere to begin',
                                 ha='center', va='center',
                                 fontdict={'fontsize': 18},
                                 transform=self.ax.transAxes)
        self.fig.show()

        cid = self.fig.canvas.mpl_connect('button_press_event', self.animate)

    def animation_init(self):
        """Initialize animation"""
        x, y = self.balls.get_xy()
        self.collection.set_color('C0')
        self.text.set_visible(False)

        return self.collection,

    def animation_update(self, i):
        self.balls.step_forward()
        x, y = self.balls.get_xy()
        for p in range(len(self.balls.balls)):
            self.patches[p].center = x[p], y[p]

        return self.collection,

    def animate(self, event):
        if not self.started:
            self.anim = animation.FuncAnimation(self.fig,
                                                self.animation_update,
                                                init_func=self.animation_init,
                                                frames=100, blit=False,
                                                interval=20)
        self.started = True
        self.fig.canvas.draw()

if __name__ == '__main__':

    mysim = sim(10, 2, v_const=2, corners=[[-3, 4], [-1, 4]],
                periodic=1)
