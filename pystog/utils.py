# -------------------------------------#
# Utilities

from collections import OrderedDict
import numpy as np

ReciprocalSpaceChoices = OrderedDict([("S(Q)", "S(Q)"),
                                      ("Q[S(Q)-1]", "also known as F(Q)"),
                                      ("FK(Q)", "Keen's F(Q)"),
                                      ("DCS(Q)", "Differential Cross-Section")])
RealSpaceChoices = OrderedDict([("g(r)", ' "little" g(r)'),
                                ("G(r)", "Pair Distribution Function"),
                                ("GK(r)", "Keen's G(r)")])

RealSpaceHeaders = ['r'] + list(RealSpaceChoices.keys())
ReciprocalSpaceHeaders = ['Q'] + list(ReciprocalSpaceChoices.keys())


def create_domain(xmin, xmax, xdelta):
    return np.arange(xmin, xmax + xdelta, xdelta)
