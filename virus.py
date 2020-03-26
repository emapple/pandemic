import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as ani
from scatter import sim
from ball import sickBallCollection


class virus(sim):
    """Class to simulate the spread of a virus in a population"""

    def __init__(self, n_ball, ndim=2, blit=True, **params):
        super().__init__(n_ball, ndim, blit=blit, ball=sickBallCollection,
                         **params)

        # assign a random ball as sick
        self.balls.balls[np.random.choice(np.arange(self.n_ball))].sick = True

        self.dotrack = params.get('dotrack', True)
        self.setup_track()

    def setup_track(self):
        """Set up plots"""
        if self.dotrack:
            self.fig3, self.ax3 = plt.subplots(1, 1)
            self.fig3.set_size_inches(6, 2)
            self.ax3.set_xticks([], [])
            self.ax3.set_yticks([], [])
            self.ax3.set_xlim([0, 400])
            self.ax3.set_ylim([0, 1])
            self.fig3.tight_layout()
            numhealthy, numsick, numcured = self.calc_track()
            self.numhealthy = [numhealthy]
            self.numsick = [numsick]
            self.numcured = [numcured]
            if self.dotrack:
                self.fills = []
                self.fills.append(self.ax3.fill_between(
                    [0], [0], self.numhealthy, facecolor='C0'))
                self.fills.append(self.ax3.fill_between([0], self.numhealthy, [sum(x) for x in
                                                                               zip(self.numhealthy, self.numsick)],
                                                        facecolor='C7'))
                self.fills.append(self.ax3.fill_between([0], [sum(x) for x in
                                                              zip(self.numhealthy, self.numsick)], [1], facecolor='C2'))
                # self.lines = [self.ax3.plot([0], x, c=color, lw=3)[0]
                #               for x, color in zip([self.numhealthy,
                #                                    self.numsick,
                #                                    self.numcured],
                #                                   ['C0', 'C7', 'C2'])]
            self.fig3.show()

    def calc_track(self):
        """Calculate fraction of healthy, sick, and cured"""
        numsick = (self.balls._getall('sick') > 0).sum()
        numcured = self.balls._getall('cured').sum()
        numhealthy = len(self.balls.balls) - numsick - numcured
        return (numhealthy / len(self.balls.balls),
                numsick / len(self.balls.balls),
                numcured / len(self.balls.balls))

    # def animation_init(self):
    #     super().animation_init()
    #     numhealthy, numsick, numcured = self.calc_trac()
    #     self.numhealthy = [numhealthy]
    #     self.numsick = [numsick]
    #     self.numcured = [numcured]
    #     if self.track:
    #         self.lines = [self.ax3.plot([0], x, color)
    #                       for x in zip([self.numhealthy,
    #                                     self.numsick, self.numcured],
    #                                    ['C0', 'C7', 'C2'])]
    #     return self.lines

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
        #   self.patches[p].set_color('C3')

        self.time += dt

        if self.blit:
            self.title.set_text(f't={self.time:.2f}')
            return self.collection, self.title
        else:
            self.ax.set_title(f't={self.time:.2f}')
            return self.collection

    def track_animation(self, i):
        numhealthy, numsick, numcured = self.calc_track()
        self.numhealthy.append(numhealthy)
        self.numsick.append(numsick)
        self.numcured.append(numsick)
        if len(self.numhealthy) > 500:
            self.numhealthy = self.numhealthy[-500:]
            self.numsick = self.numsick[-500:]
            self.numcured = self.numcured[-500:]
        self.ax3.collections.clear()
        self.fills = []
        self.fills.append(self.ax3.fill_between(
            np.arange(len(self.numhealthy[-400:])
                      ), [0] * len(self.numhealthy[-400:]),
            self.numhealthy[-400:], facecolor='C0'))
        self.fills.append(self.ax3.fill_between(np.arange(len(self.numhealthy[-400:])),
                                                self.numhealthy[-400:], [sum(x) for x in
                                                                  zip(self.numhealthy[-400:], self.numsick[-400:])],
                                                facecolor='C7'))
        self.fills.append(self.ax3.fill_between(np.arange(len(self.numhealthy[-400:])),
                                                [sum(x) for x in
                                                 zip(self.numhealthy[-400:], self.numsick[-400:])],
                                                [1] * len(self.numhealthy[-400:]), facecolor='C2'))
        # for line, numtype in zip(self.lines,
        #                          [self.numhealthy, self.numsick, self.numcured]):
        #     line.set_data(np.arange(len(numtype[-400:])), numtype[-400:])

        return self.fills

    def animate(self, event):
        super().animate(event)
        if self.dotrack:
            self.anim3 = ani.FuncAnimation(self.fig3,
                                           self.track_animation,
                                           frames=100, blit=False,
                                           interval=self.interval)
            self.fig3.canvas.draw()

if __name__ == '__main__':

    pan = virus(50, 2, radius=0.01, dt=0.005, interval=40, periodic=1,
                incubation=2, duration=4)
