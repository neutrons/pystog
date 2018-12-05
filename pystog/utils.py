# -------------------------------------#
# Utilities

import numpy as np

ReciprocalSpaceChoices = {"S(Q)": "S(Q)",
                          "Q[S(Q)-1]": "=Q[S(Q) - 1]",
                          "FK(Q)": "Keen's F(Q)",
                          "DCS(Q)": "Differential Cross-Section"}
RealSpaceChoices = {"g(r)": ' "little" g(r)',
                    "G(r)": "Pair Distribution Function",
                    "GK(r)": "Keen's G(r)"}


def create_domain(xmin, xmax, xdelta):
    return np.arange(xmin, xmax + xdelta, xdelta)
