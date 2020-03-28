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
                 radius=None, **kwargs):
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

        if radius is None:
            self.size = 0.05
        else:
            self.size = radius

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
        else:
            self.reflect()

    def reflect(self):
        """Ammend pos and vel of ball that has passed boundary

        ball1 is the ball that has crossed the boundary
        bdry_crossings are the dimensions in which it has crossed
        """
        for dim in range(self.dim):
            if (self.pos[dim] - self.size > 0
                    and self.pos[dim] + self.size < self.corners[dim][1]):
                continue
            neg_dist = self.pos[dim] - self.size
            if neg_dist <= 0:
                dist_since_crossing = neg_dist
            else:
                dist_since_crossing = (self.pos[dim] + self.size
                                       - self.corners[dim][1])

            self.pos[dim] -= dist_since_crossing
            self.vel[dim] *= -1

    @property
    def v_mag(self):
        return np.sqrt(np.sum(self.vel**2))


class sickBall(ball):
    """A ball that can get sick"""

    def __init__(self, pos=None, vel=None, corners=None, periodic=0,
                 radius=None, **kwargs):
        super().__init__(pos=pos, vel=vel, corners=corners,
                         periodic=periodic, radius=radius, **kwargs)
        self.sick = False
        self.exposed = False
        self.cured = False
        self.incubation = kwargs.get('incubation', 5)
        self.duration = kwargs.get('duration', 10)

    @property
    def sick(self):
        return self._sick

    @sick.setter
    def sick(self, status):
        if status and type(status) == bool:
            self._sick = 1.e-10  # start it close to 0
        else:
            self._sick = status

    @property
    def exposed(self):
        return self._exposed

    @exposed.setter
    def exposed(self, status):
        if status and type(status) == bool:
            self._exposed = 1.e-10  # start it close to 0
        else:
            self._exposed = status

    def advance(self, dt):
        super().advance(dt)
        if self.sick:
            self.sick += dt
            if self.sick >= self.duration:
                self.sick = False
                self.cured = True
        elif self.exposed:
            self.exposed += dt
            if self.exposed >= self.incubation:
                self.exposed = False
                self.sick = True


class ballCollection:
    """A collection of balls, including movement laws"""

    def __init__(self, n_ball, ndim, **params):
        self.n_ball = n_ball
        assert self.n_ball > 0

        self.ndim = ndim

        self.periodic = params.get('periodic', 1)
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
            self.size = params['radius']

        max_span = np.sqrt(np.sum([bdry[1]**2 for bdry in self.corners]))
        if (n_ball)**(1. / self.ndim) * self.size * 2 > 0.33 * max_span:
            raise RuntimeError('Not enough space for requested parameters\n' +
                               'Either reduce number or size of balls,' +
                               ' or increase box size')

        self.balls = [ball(vel=v, corners=self.corners, iord=i,
                           radius=self.size, **params)
                      for i, v in enumerate(vec)]

        self.time = 0

        # make sure no balls are overlapping at start
        print('Moving any initially overlapping balls')
        while(True):
            # print('iterating')
            locs1 = self.overlaps(self._getall('pos').transpose())
            if self.periodic:
                allpos2 = np.vstack([(self._getall('pos')[:, i] + 1) % bdry[1]
                                     for i, bdry in enumerate(self.corners)])
                locs2 = self.overlaps(allpos2)
            else:
                locs2 = ()

            if len(locs1) == 0 and len(locs2) == 0:
                break
            for i in locs1 + [loc for loc in locs2 if loc not in locs1]:
                self.balls[i[0]].pos = self.balls[i[0]].pos + 4 * \
                    (0.5 - np.random.random(self.ndim)) * self.size
                self.balls[i[0]].wrap()

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
        """Step forward all balls in ballCollection"""
        for ball in self.balls:
            ball.advance(dt)

    def filt_pos_vel(self, iords=None):
        """Return pos and vel of balls with iord in iords"""
        allpos = self._getall('pos')
        allvel = self._getall('vel')
        if iords is not None:
            # this code gets the indices that have iord in iords
            iords = np.array(iords)
            idx = np.argmax(self._getall('iord')[
                None, :] == iords[:, None], axis=1)
            allpos = allpos[idx]
            allvel = allvel[idx]

        return allpos, allvel

    def overlaps(self, positions):
        """Calculate which balls are overlapping"""
        dp = np.array([np.subtract.outer(p, p)
                       for p in positions])
        pp = (dp * dp).sum(axis=0) - (2 * self.size)**2

        # exclude diagonals
        locs = np.where((pp < 0) & (np.eye(len(pp)) != 1))

        temp_locs = [(i, j) for i, j in zip(locs[0], locs[1])]

        return list(set([tuple(self.pair_sort(loc)) for loc in temp_locs]))

    @staticmethod
    def pair_sort(pair):
        """Sorts a 2-tuple"""
        if pair[0] > pair[1]:
            return (pair[1], pair[0])
        else:
            return pair


class hardBallCollection(ballCollection):
    """Hard scattering ball"""

    def time_to_collision(self, positions, velocities):
        """Calculate how long until balls collide

        Calculates for all pairwise combinations
        Balls that will not collide are nans
        """
        dp = np.array([np.subtract.outer(p, p)
                       for p in positions])
        dv = np.array([np.subtract.outer(v, v)
                       for v in velocities])
        pp = (dp * dp).sum(axis=0) - (2 * self.size)**2
        pv = (dp * dv).sum(axis=0)
        vv = (dv * dv).sum(axis=0)

        return (-pv - np.sqrt((pv * pv) - (pp * vv))) / vv

    def will_collide(self, dt, iords=None):
        """Calculate pairs that will collide within time dt

        Returns indices, not iords
        """
        allpos, allvel = self.filt_pos_vel(iords=iords)

        T = self.time_to_collision(allpos.transpose(), allvel.transpose())
        locs1 = np.where(((T > 0) & (T <= dt)))

        if self.periodic:
            allpos2 = np.vstack([(allpos[:, i] + 1) % bdry[1]
                                 for i, bdry in enumerate(self.corners)])
            T2 = self.time_to_collision(allpos2, allvel.transpose())
            locs2 = np.where(((T2 > 0) & (T2 <= dt)))
        else:
            locs2 = []

        temp_loc = [(i, j) for i, j in zip(locs1[0], locs1[1])]
        locs1 = list(set([self.pair_sort(pair) for pair in temp_loc]))

        if self.periodic:
            temp_loc2 = [(i, j) for i, j in zip(locs2[0], locs2[1])
                         if (i, j) not in temp_loc]
            locs2 = list(set([self.pair_sort(pair) for pair in temp_loc2]))

        t_to_intersection = [T[idx]
                             for idx in locs1] + [T2[idx] for idx in locs2]

        locs = locs1 + locs2

        assert len(t_to_intersection) == len(locs)
        return locs, t_to_intersection

    def step_forward(self, dt, multistep=True, iords=None):
        """Step forward all balls in hardBallCollection

        Override of method in ballCollection
        If multistep, will subdivide to avoid multiple simultaneous
            collisions
        If iords are passed, only calculates for subset of balls with
        iord in iords
        """
        collij, t_to_intersect = self.will_collide(dt, iords=iords)
        idx_list = np.unique(collij)

        if (multistep and
                len(collij) > 0 and
                np.max(np.unique(collij, return_counts=True)[1])) > 1:
            # print('Recurse')
            if iords is None:
                iords = np.arange(len(self.balls))
            for i in range(10):
                self.step_forward(dt / 10, multistep=True,
                                  iords=iords[idx_list])
                self.cleanup(dt)
        else:
            if iords is not None:
                balls_to_advance = [ball for i, ball in enumerate(self.balls)
                                    if ball.iord in iords]
            else:
                balls_to_advance = self.balls

            [ball.advance(dt)
             for i, ball in enumerate(balls_to_advance) if i not in idx_list]

            self.time += dt

            # avoiding repeats
            if len(collij) > 0:
                for loc, tto in zip(collij, t_to_intersect):
                    balls_to_advance[loc[0]].advance(tto)
                    balls_to_advance[loc[1]].advance(tto)
                    # they should now be bordering each other
                    # print(np.sqrt(np.sum((self.balls[loc[0]].pos -
                    # self.balls[loc[1]].pos)**2)))
                    # assert(abs(np.sqrt(np.sum((self.balls[loc[0]].pos -
                    # self.balls[loc[1]].pos)**2)) - (2 * self.size)) < 1e-5)
                    self.collide(balls_to_advance[
                                 loc[0]], balls_to_advance[loc[1]])
                    balls_to_advance[loc[0]].advance(dt - tto)
                    balls_to_advance[loc[1]].advance(dt - tto)
            self.cleanup(dt)

    def collide(self, ball1, ball2):
        """Ammend velocities of colliding balls

        Does not change any positions, so balls should be at the point
        of collision
        """
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

        # assert momentum and energy conserved
        assert np.all((ball1.vel + ball2.vel - (vel1old + vel2old)) < 1e-5)
        assert np.sum(ball1.vel**2 + ball2.vel**2) - \
            np.sum(vel1old**2 + vel2old**2) < 1e-5

    def cleanup(self, dt):
        """Fix any balls that managed to overlap

        When predicting collisions in will_collide(), we do not
        account for possible collisions *after turnaround*, which is
        where most of the accidental overlaps occur. Also, some edge
        cases need correction.
        """

        locs = self.overlaps(self._getall('pos').transpose())
        if self.periodic:
            allpos2 = np.vstack([(self._getall('pos')[:, i] + 1) % bdry[1]
                                 for i, bdry in enumerate(self.corners)])
            locs2 = self.overlaps(allpos2)
        else:
            locs2 = []

        locs += [loc for loc in locs2 if loc not in locs]
        for pair in locs:
            relative_vel = np.sqrt(np.sum((self.balls[pair[0]].vel
                                           - self.balls[pair[1]].vel)**2))
            relative_pos = np.sqrt(np.sum([min(((self.balls[pair[0]].pos[i] -
                                                 self.balls[pair[1]].pos[i])
                                                % bdry[1]),
                                               ((self.balls[pair[1]].pos[i] -
                                                 self.balls[pair[0]].pos[i])
                                                % bdry[1]))**2
                                           for i, bdry in enumerate(self.corners)]))
            # print(relative_pos)
            missed_time = abs(relative_pos - 2 * self.size) / relative_vel

            for i in [0, 1]:
                self.balls[pair[i]].advance(-missed_time)
            self.collide(self.balls[pair[0]], self.balls[pair[1]])
            for i in [0, 1]:
                self.balls[pair[i]].advance(missed_time)

            # assert missed_time < dt
            # print(missed_time, dt)


class sickBallCollection(hardBallCollection):
    """Ball collection capable of getting sick"""

    def __init__(self, n_ball, ndim, **params):
        super().__init__(n_ball, ndim, **params)
        self.incubation = params.get('incubation', 5)
        self.duration = params.get('duration', 10)

        # overwriting balls because we want them to be sickBalls

        self.balls = [sickBall(pos=ball.pos, vel=ball.vel,
                               corners=ball.corners, iord=ball.iord,
                               periodic=ball.periodic,
                               radius=ball.size, incubation=self.incubation,
                               duration=self.duration) for ball in self.balls]

    def collide(self, ball1, ball2):
        super().collide(ball1, ball2)
        if ball1.sick and self.can_catch(ball2):
            ball2.exposed = 1.e-10
        elif self.can_catch(ball1) and ball2.sick:
            ball1.exposed = 1.e-10

    @staticmethod
    def can_catch(ball):
        if not (ball.sick or ball.exposed or ball.cured):
            return True
        else:
            return False
