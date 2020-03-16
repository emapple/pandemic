import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from matplotlib import animation


class DimensionError(Exception):
    pass


class ball:
    """A billiard ball object

    Contains 3D space and 3D velocities
    """

    def __init__(self, pos=None, vel=None, corners=None, periodic=0,
                 **kwargs):
        """Initialize 6D phase space of ball

        If pos not provided, randomly chosen positions according to
                corners, which must be provided

        corners is a series of tuples/lists defining boundaries,
                        e.g. ((minx, maxx), (miny, maxy))
        """

        self.corners = corners
        self.periodic = periodic

        if pos is None:
            if corners is None:
                raise ValueError('Must provide either pos or corners')
            else:
                self.dim = len(corners)
                if self.dim > 3:
                    raise DimensionError('Dimenions grater than 3 '
                                         'not accepted')
                lims = np.array(corners)
                self.pos = (np.random.random(size=self.dim)
                            * abs(lims[:, 1] - lims[:, 0])
                            + lims[:, 0])

        else:
            if len(pos) > 3:
                raise DimensionError('Dimenions grater than 3 not accepted')
            else:
                self.dim = len(pos)
            self.pos = np.array(pos)

        if vel is not None:
            assert len(self.pos) == len(vel)
            self.vel = np.array(vel)
        else:
            self.vel = np.array([0] * len(self.pos))

        if periodic:
            if self.corners is None:
                raise ValueError('Must specify boundaries for periodic'
                                 'boundary conditions')
            self.periodic = True

    def advance(self, dt):
        """Advance position forward by time dt"""
        self.pos += self.vel * dt
        if self.periodic:
            for i, bdry in enumerate(self.corners):
                self.pos[i] = (self.pos[i] - bdry[0]) % bdry[1]
                if self.pos[i] < bdry[0]:
                    self.pos[i] = self.pos[i] + bdry[0]

    @property
    def v_mag(self):
        return np.sqrt(np.sum(self.vel**2))


class sim:
    """Simulation object"""

    def __init__(self, n_ball, ndim, **params):

        self.n_ball = n_ball
        assert self.n_ball > 0
        
        self.ndim = ndim

        self.corners = params.get(
            'corners', [[0, 1] for i in range(self.ndim)])

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
        else:
            self.ax.set_ylim([-1, 1])
            self.fig.set_size_inches(5, 1)

        self.scat = self.ax.scatter([], [])

        self.anim = animation.FuncAnimation(self.fig, self.animation_update,
                                            init_func=self.animation_init,
                                            frames=100, blit=False,
                                            interval=20)

    def _getall(self, attr):
        """Returns attribute attr for all balls"""

        return np.array([x.__getattribute__(attr) for x in self.balls])

    def animation_init(self):
        """Initialize animation"""
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
        return self.scat,

    def step_forward(self):
        for ball in self.balls:
            ball.advance(0.01)


if __name__ == '__main__':

    bill = ball(corners=[[-1, 1], [-3, 3]])
    mysim = sim(3, 2, v_const=2)
