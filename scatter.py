import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from matplotlib import animation as ani
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

        if 'ball' in params:
            if params['ball'].lower().startswith('hard'):
                self.balls = hardBallCollection(
                    self.n_ball, self.ndim, **params)
        else:
            self.balls = ballCollection(self.n_ball, self.ndim, **params)

        self.dt = params.get('dt', 0.01)
        self.interval = params.get('interval', 20)

        self.time = 0.0

        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(self.balls.corners[0])
        if self.ndim > 1:
            self.ax.set_ylim(self.balls.corners[1])
            xspan = float(np.diff(self.balls.corners[0]))
            yspan = float(np.diff(self.balls.corners[1]))

            scaler = 8. / xspan
            xspan *= scaler
            yspan *= scaler
            self.fig.set_size_inches(xspan, yspan)
            self.ax.set_aspect('equal')
        else:
            xspan = float(np.diff(self.balls.corners[0]))
            yspan = xspan / 10
            self.ax.set_ylim([-yspan/2, yspan/2])
            self.fig.set_size_inches(8, 1)
            self.ax.set_aspect('equal')

        self.ax.set_title(f't={self.time:.2f}')
        self.fig.tight_layout()

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

    def animation_update(self, i, dt):
        self.balls.step_forward(dt)
        x, y = self.balls.get_xy()
        for p in range(len(self.balls.balls)):
            self.patches[p].center = x[p], y[p]

        self.time += dt
        self.ax.set_title(f't={self.time:.2f}')

        return self.collection,

    def animate(self, event):
        if not self.started:
            self.anim = ani.FuncAnimation(self.fig,
                                          (lambda i:
                                           self.animation_update(i,
                                                                 self.dt)),
                                          init_func=self.animation_init,
                                          frames=100, blit=False,
                                          interval=self.interval)
        self.started = True
        self.fig.canvas.draw()

if __name__ == '__main__':

    mysim = sim(10, 2, v_const=2, corners=[[-3, 4], [-1, 4]],
                periodic=1)
