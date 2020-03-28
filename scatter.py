import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from matplotlib import animation as ani
from matplotlib import collections as clt
from ball import ball, ballCollection, hardBallCollection


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

    def __init__(self, n_ball, ndim, blit=True, **params):

        self.ndim = ndim
        self.n_ball = n_ball

        if 'ball' in params:  # this should be a class
            self.balls = params['ball'](self.n_ball, self.ndim, **params)
        else:
            self.balls = hardBallCollection(self.n_ball, self.ndim, **params)

        self.dt = params.get('dt', 0.01)
        self.interval = params.get('interval', 40)
        self.dohist = params.get('dohist', False)
        self.blit = blit
        if self.blit:
            print('Warning: blitting improves performance but causes issues'
                  'when pausing')

        if 'colorby' in params:
            self.colorby = params['colorby']
        else:
            self.colorby = None

        if self.colorby == 'velocity':
            self.cmap = plt.cm.ScalarMappable(cmap=params.get('cmap',
                                                              'viridis'))
            self.cmap.set_clim(0, 2 * np.max(self.balls._getall('v_mag')))

        self.time = 0.0

        self.setup_plots()

    def setup_plots(self):
        """Do initial setup of plots"""
        self.fig, self.ax = plt.subplots(1, 1)
        if self.dohist:
            self.fig2, self.ax2 = plt.subplots(1, 1)
            self.fig2.set_size_inches(5, 3)
            self.ax2.set_xlabel('Velocity')
            self.ax2.set_ylabel('N')
            self.ax2.set_title(f't={self.time:.2f}')
            self.histbins = np.linspace(0,
                                        2.5 *
                                        np.max(self.balls._getall('v_mag')),
                                        max(2, len(self.balls.balls) // 9))
            vals, histbins = np.histogram(self.balls._getall('v_mag'),
                                          bins=self.histbins)
            self.ax2.set_ylim([0, 1.2 * np.max(vals)])
            self.ax2.set_xlim([0, 1.2 * histbins[1:][vals > 0][-1]])
            self.barcollection = self.ax2.bar(0.5 * (histbins[1:]
                                                     + histbins[:-1]), vals,
                                              np.diff(histbins)[0], color='C0')
            self.fig2.tight_layout()
            self.fig2.show()
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
            self.ax.set_ylim([-yspan / 2, yspan / 2])
            self.fig.set_size_inches(8, 1)
            self.ax.set_aspect('equal')

        if self.blit:
            self.title = self.ax.text(np.mean(self.balls.corners[0]),
                                      self.ax.get_ylim()[-1] * 0.95,
                                      f't={self.time:.2f}')
        else:
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
        if self.colorby == 'velocity':
            self.collection.set_color([self.cmap.to_rgba(v)
                                       for v in self.balls._getall('v_mag')])

        if self.blit:
            return self.collection, self.title
        else:
            return self.collection

    def animation_update(self, i, dt):
        self.balls.step_forward(dt)
        x, y = self.balls.get_xy()
        for p in range(len(self.balls.balls)):
            self.patches[p].center = x[p], y[p]

        self.time += dt

        if self.colorby == 'velocity':
            self.collection.set_color([self.cmap.to_rgba(v)
                                       for v in self.balls._getall('v_mag')])

        if self.blit:
            self.title.set_text(f't={self.time:.2f}')
            return self.collection, self.title
        else:
            self.ax.set_title(f't={self.time:.2f}')
            return self.collection

    def animate(self, event):
        if not self.started:
            self.anim = ani.FuncAnimation(self.fig,
                                          (lambda i:
                                           self.animation_update(i,
                                                                 self.dt)),
                                          init_func=self.animation_init,
                                          frames=100, blit=self.blit,
                                          interval=self.interval)
            self.started = True
            self.pause = False
        else:
            if self.pause:
                self.anim.event_source.start()
                self.pause = False
            else:
                self.anim.event_source.stop()
                self.pause = True
        self.fig.canvas.draw()
        if self.dohist:
            self.anim2 = ani.FuncAnimation(self.fig2,
                                           self.hist_animation,
                                           frames=100, blit=False,
                                           interval=self.interval)
            self.fig2.canvas.draw()

    def hist_animation(self, i):
        vals, histbins = np.histogram(self.balls._getall('v_mag'),
                                      bins=self.histbins)
        [bc.set_height(v) for bc, v in zip(self.barcollection, vals)]
        self.ax2.set_title(f't={self.time:.2f}')
        self.ax2.set_ylim([0, np.max(vals) * 1.2])
        self.ax2.set_xlim([0, 1.2 * histbins[1:][vals > 0][-1]])

        return self.barcollection,


if __name__ == '__main__':

    # mysim = sim(10, 2, v_const=2, corners=[[-3, 4], [-1, 4]],
                # periodic=1)

    mysim = sim(50, 2, v_maxwell_mu=5, v_maxwell_sigma=1, periodic=0,
                corners=[[-3, 3], [-3, 3]], dt=0.005, rad=0.1,
                interval=40, dohist=True)
