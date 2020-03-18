import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from matplotlib import animation
from ball import *


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

        self.scat = self.ax.scatter([], [])
        self.size_in_points = self.ax.transData.transform(self.size)
        self.started = False
        self.text = self.ax.text(0.5, 0.5, 'Click anywhere to begin',
                                 ha='center', va='center',
                                 fontdict={'fontsize': 18},
                                 transform=self.ax.transAxes)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.animate)

    def _getall(self, attr):
        """Returns attribute attr for all balls"""

        return np.array([x.__getattribute__(attr) for x in self.balls])

    def animation_init(self):
        """Initialize animation"""
        self.text.set_visible(False)
        self.scat.remove()
        self.scat = self.ax.scatter([], [])
        return self.scat,

    def animation_update(self, i):
        self.step_forward()
        x = self._getall('pos')[:, 0]
        if self.ndim > 1:
            y = self._getall('pos')[:, 1]
        else:
            y = np.zeros(len(x))
        self.scat.set_offsets([[ix, iy] for ix, iy in zip(x, y)])
        self.scat.set_array(np.zeros(len(x)))
        self.scat.set_sizes(np.ones(len(x)) * self.size_in_points**2)
        return self.scat,

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
