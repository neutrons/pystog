import numpy as np
import pandas as pd

# -------------------------------------#
# Utilities

ReciprocalSpaceChoices = {"S(Q)": "S(Q)",
                          "Q[S(Q)-1]": "=Q[S(Q) - 1]",
                          "FK(Q)": "Keen's F(Q)",
                          "DCS(Q)": "Differential Cross-Section"}
RealSpaceChoices = {"g(r)": ' "little" g(r)',
                    "G(r)": "Pair Distribution Function",
                    "GK(r)": "Keen's G(r)"}


def get_data(filename, skiprows=0, skipfooter=0, xcol=0, ycol=1):
    # Setup domain
    return pd.read_csv(filename, sep=r"\s*",
                       skiprows=skiprows,
                       skipfooter=skipfooter,
                       usecols=[xcol, ycol],
                       names=['x', 'y'],
                       engine='python')


def create_domain(xmin, xmax, binsize):
    x = np.arange(xmin, xmax + binsize, binsize)
    return x


def write_out(filename, xdata, ydata, title=None):
    with open(filename, 'w') as f:
        f.write("%d \n" % len(xdata))
        f.write("%s \n" % title)
        for x, y in zip(xdata, ydata):
            f.write('%f %f\n' % (x, y))
