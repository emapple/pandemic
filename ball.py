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

        self.iord = kwargs.get('iord', None)

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

    def __str__(self):
        return f'ball {self.iord}, pos={self.pos}, vel={self.vel}'

    def advance(self, dt):
        """Advance position forward by time dt"""
        self.pos += self.vel * dt
        self.wrap()

    def wrap(self):
        if self.periodic:
            for i, bdry in enumerate(self.corners):
                self.pos[i] %= bdry[1]

    @property
    def v_mag(self):
        return np.sqrt(np.sum(self.vel**2))


class ballCollection:
    """A collection of balls, including movement laws"""

    def __init__(self, n_ball, ndim, **params):
        self.n_ball = n_ball
        assert self.n_ball > 0

        self.ndim = ndim

        if 'corners' in params:
            # move origin to 0
            self.corners = [[0, np.diff(corner)[0]]
                            for corner in params['corners']]
            params.pop('corners', None)
        else:
            self.corners = [[0, 1] for i in range(self.ndim)]

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

        self.balls = [ball(vel=v, corners=self.corners, iord=i, **params)
                      for i, v in enumerate(vec)]

        self.time = 0

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

    def _setall(self, attr, all_attrs):
        """Setts attribute attr for all balls"""
        [ball.__setattr__(attr, val)
         for ball, val in zip(self.balls, all_attrs)]

    def step_forward(self, dt):
        for ball in self.balls:
            ball.advance(dt)


class hardBallCollection(ballCollection):
    """Hard scattering ball"""

    def will_collide(self, dt, iords=None):
        """Calculate pairs that will collide within time dt

        Returns indices, not iords
        """
        allpos = self._getall('pos')
        allvel = self._getall('vel')
        if iords is not None:
            # this code gets the indices that have iord in iords
            iords = np.array(iords)
            idx = np.argmax(self._getall('iord')[
                            None, :] == iords[:, None], axis=1)
            allpos = allpos[idx]
            allvel = allvel[idx]
        dp = np.array([np.subtract.outer(p, p)
                       for p in allpos.transpose()])
        allpos2 = np.vstack([(allpos[:, i] + 1) % bdry[1]
                             for i, bdry in enumerate(self.corners)])
        dp2 = np.array([np.subtract.outer(p, p)
                        for p in allpos2])
        dv = np.array([np.subtract.outer(v, v)
                       for v in allvel.transpose()])
        pp = (dp * dp).sum(axis=0) - (2 * self.size)**2
        pp2 = (dp2 * dp2).sum(axis=0) - (2 * self.size)**2
        pv = (dp * dv).sum(axis=0)
        pv2 = (dp2 * dv).sum(axis=0)
        vv = (dv * dv).sum(axis=0)
        T = (-pv - np.sqrt((pv * pv) - (pp * vv))) / vv
        T2 = (-pv2 - np.sqrt((pv2 * pv2) - (pp2 * vv))) / vv

        locs1 = np.where(((T > 0) & (T <= dt)))
        locs2 = np.where(((T2 > 0) & (T2 <= dt)))

        temp_loc = [(i, j) for i, j in zip(locs1[0], locs1[1])]
        t2 = [T2[i, j] for i, j in zip(locs2[0], locs2[1])
              if (i, j) not in temp_loc]
        temp_loc = temp_loc + [(i, j) for i, j in zip(locs2[0], locs2[1])
                               if (i, j) not in temp_loc]

        locs = np.array(temp_loc).reshape((2, len(temp_loc))).astype(int)

        if len(locs[0]) != len(locs1[0]):
            print('Edge effect found\n\n\n')

        t_to_intersection = np.array(list(T[locs1]) + t2)
        assert len(t_to_intersection) == len(locs[0])
        return locs, t_to_intersection

    def step_forward(self, dt, multistep=True, iords=None):
        """If multistep, will subdivide to avoid multiple simultaneous collisions"""
        (colli, collj), t_to_intersect = self.will_collide(dt, iords=iords)
        if (multistep and
                len(colli) > 0 and
                np.max(np.unique(colli, return_counts=True)[1])) > 1:
            if iords is None:
                iords = np.arange(len(self.balls))
            for i in range(10):
                self.step_forward(dt / 10, multistep=True,
                                  iords=iords[np.sort(list(set(colli)))])
        else:
            if iords is not None:
                balls_to_advance = [ball for i, ball in enumerate(self.balls)
                                    if ball.iord in iords]
            else:
                balls_to_advance = self.balls
                # for i, ball in enumerate(self.balls):
            [ball.advance(dt)
             for i, ball in enumerate(balls_to_advance) if i not in colli]

            self.time += dt
            # if i not in colli:
            #     ball.advance(dt)
            # avoiding repeats
            if len(colli) > 0:
                for i, j, tto in zip(colli[:len(colli) // 2], collj[:len(collj) // 2],
                                     t_to_intersect[:len(t_to_intersect // 2)]):
                    balls_to_advance[i].advance(tto)
                    balls_to_advance[j].advance(tto)
                    # they should now be bordering each other
                    # assert(np.sqrt(np.sum((self.balls[i].pos - self.balls[j].pos)**2))
                    #        - (2 * self.size) < 1e-5)
                    self.collide(balls_to_advance[i], balls_to_advance[j])
                    balls_to_advance[i].advance(dt - tto)
                    balls_to_advance[j].advance(dt - tto)

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
