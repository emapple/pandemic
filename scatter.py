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

        self.n_ball = n_ball
        assert self.n_ball > 0

        self.ndim = ndim

        self.corners = params.get(
            'corners', [[0, 1] for i in range(self.ndim)])
        params.pop('corners', None)

        if 'v_const' in params:
            v_init = np.array([params['v_const']] * self.n_ball)
        elif 'v_maxwell_mu' in params:
            if 'v_maxwell_sigma' not in params:
                raise ValueError('Must provide both mu and scale for params')
            mx = maxwell(loc=params['v_maxwell_mu'],
                         scale=params['v_maxwell_sigma'])
            v_init = mx.rvs(self.n_ball)
        else:
            v_init = np.random.random(size=self.n_ball)

        vec = np.random.multivariate_normal(np.zeros(self.ndim),
                                            cov=np.eye(self.ndim),
                                            size=self.n_ball)

        vec = (vec / np.sqrt(np.sum(vec**2, axis=1)).reshape(len(vec), 1)
               * v_init.reshape(len(v_init), 1))

        if 'rad' in params:
            self.size = params['rad']
        elif 'radius' in params:
            self.size = params['radius']
        else:
            params['radius'] = 0.05
            self.size = params['radius']

        self.balls = [ball(vel=v, corners=self.corners, **params) for v in vec]

        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(self.corners[0])
        if self.ndim > 1:
            self.ax.set_ylim(self.corners[1])
            xspan = float(np.diff(self.corners[0]))
            yspan = float(np.diff(self.corners[1]))

            scaler = 5. / xspan
            xspan *= scaler
            yspan *= scaler
            self.fig.set_size_inches(xspan, yspan)
            self.ax.set_aspect('equal')
        else:
            self.ax.set_ylim([-1, 1])
            self.fig.set_size_inches(5, 1)

        x, y = self._get_xy()
        self.patches = [plt.Circle((ix, iy), self.size)
                        for ix, iy in zip(x, y)]
        self.collection = UpdatablePatchCollection(
            self.patches, edgecolor='none', facecolor='none')
        self.collection.set_array(np.zeros(len(self.balls)))
        self.ax.add_collection(self.collection)
        self.started = False
        self.text = self.ax.text(0.5, 0.5, 'Click anywhere to begin',
                                 ha='center', va='center',
                                 fontdict={'fontsize': 18},
                                 transform=self.ax.transAxes)
        self.fig.show()

        cid = self.fig.canvas.mpl_connect('button_press_event', self.animate)

    def _getall(self, attr):
        """Returns attribute attr for all balls"""
        return np.array([x.__getattribute__(attr) for x in self.balls])

    def _get_xy(self):
        """Convenience function for getting just x and y values"""
        x = self._getall('pos')[:, 0]
        if self.ndim > 1:
            y = self._getall('pos')[:, 1]
        else:
            y = np.zeros(len(x))
        return (x, y)

    def animation_init(self):
        """Initialize animation"""
        x, y = self._get_xy()
        self.collection.set_color('C0')
        self.text.set_visible(False)

        return self.collection,

    def animation_update(self, i):
        self.step_forward()
        x, y = self._get_xy()
        for p in range(len(self.balls)):
            self.patches[p].center = x[p], y[p]

        return self.collection,

    def step_forward(self):
        for ball in self.balls:
            ball.advance(0.01)

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
