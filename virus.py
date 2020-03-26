import numpy as np
import matplotlib.pyplot as plt
from scatter import sim
from ball import sickBallCollection


class virus(sim):
    """Class to simulate the spread of a virus in a population"""

    def __init__(self, n_ball, ndim=2, blit=True, **params):
        super().__init__(n_ball, ndim, blit=blit, ball=sickBallCollection,
                         **params)

        # assign a random ball as sick
        self.balls.balls[np.random.choice(np.arange(self.n_ball))].sick = True

    def animation_update(self, i, dt):
        self.balls.step_forward(dt)
        x, y = self.balls.get_xy()
        for p in range(len(self.balls.balls)):
            self.patches[p].center = x[p], y[p]
        self.collection.set_color(['C7' if ball.sick else
                                   'C1' if ball.exposed
                                   else 'C2' if ball.cured
                                   else 'C0' for ball in self.balls.balls])
        # 'C0' if not ball.sick else 'C7'
        #                           for ball in self.balls.balls])
        # if self.balls.balls[p].sick:
        # 	self.patches[p].set_color('C3')

        self.time += dt

        if self.blit:
            self.title.set_text(f't={self.time:.2f}')
            return self.collection, self.title
        else:
            self.ax.set_title(f't={self.time:.2f}')
            return self.collection

if __name__ == '__main__':

    pan = virus(50, 2, radius=0.01, dt=0.005, interval=40, periodic=1,
                incubation=2, duration=4)
