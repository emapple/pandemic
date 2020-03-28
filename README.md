# pandemic

A basic pandemic simulation, inspired by the [Washington Post's simulations](https://www.washingtonpost.com/graphics/2020/world/corona-simulator/?itid=sf_coronavirus). Also works as a plain hard sphere scattering simulation.

This should be simple to run, and the default parameters should be somewhat sensible.

## Scattering simulation

To run a scattering simulation (in `scatter.py`), run
`python scatter.py` at the command line, which will run with a particular set of parameters,
at the command line
or
```
from scatter import sim
sim(50, 2, **kwargs)
```
from within a python session. The possible keyword arguments are listed below.

The required arguments for `sim` are the number of particles and the number of dimensions (1, 2, or 3). All units are abritrary, so what matters is fiddling with parameters and comparing outcomes, not necessarily the outcome of a single simulation. The 3 dimensional case has not been tested, and is less interesting as the visualization is only 2D anyway.

By default, a histogram of the velocity magnitudes will also be plotted, so you can see how the equilibrium always approaches something like the [Maxwell-Boltzmann distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution). You can turn off the histogram with `dohist=False`.

Optional parameters include `periodic` (default False), which determines whether there are periodic boundary conditions or if balls will scatter off walls. We also have `v_const`, which assigns an initial speed to all balls. By default, they all start with `v_const=1`. `v_maxwell_mu` and `v_maxwell_sigma` can be specified instead to initialize from a Maxwell distribution. The boxsize is by default 2X2. You may specify the box with the `corners` argument, which may be of the form `[[xi, xj], [yi, yj]]` or `[side1, side2]`. In any case, the box will be translated so one corner is at the origin. The radius of the balls may be specified with `radius` (default about 1% of box size).

If you'd like to pause and keep the balls where they are, you have to turn of blitting, which does add considerable overhead to the animation and may reduce the number of balls you can simulate at once without enormous lag. To turn off blitting, set `blit=False`.

## Pandemic simulation

This simulation is built off the regular hard sphere scattering, but simulates the spread of illness. In this simulation, the spheres represent people, who may either be healthy, infected but not yet contagious, sick/contagious, and cured. There are not yet any other outcomes. You may run `python virus.py` at the command line, or
```
from virus import virus
virus(50, **kwargs)
```
from within a python session. Many arguments are the same as above, but with potentially different defaults. Additional arguments include `incubation` (default 1) and `duration` (default 3), which define how long someone is infected but not yet contagious, and how long they are sick/contagious before healing. By default, a figure will show the numbers of people of each illness status; you may turn this off with `dotrack=False`.

## How do parameters influence outcome?
If you try the following simulations, you can note the difference in outcome, or the difference in time taken to reach a given outcome.
```
virus(50)
virus(50, v_const=0.5)  # slow everything down -- sort of social distacing
virus(50, v_const=2)  # speed everything up -- what happens if you go to a parade?
virus(100)  # look how much faster densely populated areas spread disease
virus(10)  # the opposite is true in this case
```

You can also play with the incubation period and illness duration to see what happens.

## To Do:

Many things! Social distancing measures, quarantine measures, adding "parties" and other major gathering events, etc. I add when I have time.
