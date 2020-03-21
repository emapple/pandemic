import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell

# with thanks to
# https://pdfs.semanticscholar.org/cd56/57befb9af4fd531d33892ed9e5b0098de1d6.pdf
# for some useful information


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
            if self.corners is None:
                raise ValueError('Must provide either pos or corners')
            else:
                self.dim = len(self.corners)
                if self.dim > 3:
                    raise DimensionError('Dimenions grater than 3 '
                                         'not accepted')
                lims = np.array(self.corners)
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

        if 'radius' in kwargs:
            self.size = kwargs['radius']
        else:
            self.size = 0.05

    def advance(self, dt):
        """Advance position forward by time dt"""
        self.pos += self.vel * dt
        self.wrap()

    def wrap(self):
        if self.periodic:
            for i, bdry in enumerate(self.corners):
                if self.pos[i] > bdry[1]:
                    self.pos[i] = self.pos[i] - (bdry[1] - bdry[0])
                elif self.pos[i] <= bdry[0]:
                    self.pos[i] = self.pos[i] + (bdry[1] - bdry[0])

    @property
    def v_mag(self):
        return np.sqrt(np.sum(self.vel**2))


class ballCollection:
    """A collection of balls, including movement laws"""

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
        # make sure no balls are overlapping at start
        dp = np.array([np.subtract.outer(p, p)
                       for p in self._getall('pos').transpose()])
        pp = (dp * dp).sum(axis=0) - (2 * self.size)**2
        pp[np.arange(len(pp)), np.arange(len(pp))] = 1
        locs = np.where(pp <= 0)
        if len(locs[0]) > 0:
            print('Moving initial overlapping balls')
            while(True):
                for i in locs[0][:len(locs[0]) // 2]:
                    self.balls[i].pos = self.balls[i].pos + 2 * \
                        (0.5 - np.random.random(self.ndim)) * self.size
                dp = np.array([np.subtract.outer(p, p)
                               for p in self._getall('pos').transpose()])
                pp = (dp * dp).sum(axis=0) - (2 * self.size)**2
                pp[np.arange(len(pp)), np.arange(len(pp))] = 1
                locs = np.where(pp <= 0)
                if len(locs[0]) == 0:
                    break

    def get_xy(self):
        """Convenience function for getting just x and y values"""
        x = self._getall('pos')[:, 0]
        if self.ndim > 1:
            y = self._getall('pos')[:, 1]
        else:
            y = np.zeros(len(x))
        return (x, y)

    def _getall(self, attr):
        """Returns attribute attr for all balls"""
        return np.array([x.__getattribute__(attr) for x in self.balls])

    def step_forward(self, dt):
        for ball in self.balls:
            ball.advance(dt)


class hardBallCollection(ballCollection):
    """Hard scattering ball"""

    def will_collide(self, dt):
        """Calculate pairs that will collide within time dt"""
        dp = np.array([np.subtract.outer(p, p)
                       for p in self._getall('pos').transpose()])
        dv = np.array([np.subtract.outer(v, v)
                       for v in self._getall('vel').transpose()])
        pp = (dp * dp).sum(axis=0) - (2 * self.size)**2
        pv = (dp * dv).sum(axis=0)
        vv = (dv * dv).sum(axis=0)
        T = (-pv - np.sqrt((pv * pv) - (pp * vv))) / vv
        locs1 = np.where((T > 0) & (T <= dt))

        t_to_intersection = T[locs1]
        return locs1, t_to_intersection

    def step_forward(self, dt):
        (colli, collj), t_to_intersect = self.will_collide(dt)
        for i, ball in enumerate(self.balls):
            if i not in colli:
                ball.advance(dt)
        # avoiding repeats
        if len(colli) > 0:
            for i, j, tto in zip(colli[:len(colli) // 2], collj[:len(collj) // 2],
                                 t_to_intersect[:len(t_to_intersect // 2)]):
                self.balls[i].advance(tto)
                self.balls[j].advance(tto)
                # they should now be bordering each other
                assert(np.sqrt(np.sum((self.balls[i].pos - self.balls[j].pos)**2))
                              - (2 * self.size) < 1e-5)
                self.collide(self.balls[i], self.balls[j])
                self.balls[i].advance(dt - tto)
                self.balls[j].advance(dt - tto)

    def collide(self, ball1, ball2):
        """Ammend velocities of colliding balls"""
        normal = ball2.pos - ball1.pos
        normal = normal / np.sqrt(np.sum(normal**2))
        if self.ndim == 1:
            tangent = np.array([0])
        else:
            tangent = np.copy(normal[::-1])
            tangent[0] = -tangent[0]

        vel1old = np.copy(ball1.vel)
        vel2old = np.copy(ball2.vel)

        ball1_vel_temp = normal * (normal.dot(ball2.vel)) + \
            tangent * (tangent.dot(ball1.vel))
        ball2.vel = normal * (normal.dot(ball1.vel)) + \
            tangent * (tangent.dot(ball2.vel))
        ball1.vel = ball1_vel_temp

        assert np.all((ball1.vel + ball2.vel - (vel1old + vel2old)) < 1e-5)
