import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell


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


# class hardball(ball):
# 	"""Hard scattering ball"""
# 	def advance(self, dt):
# 		collisions =
