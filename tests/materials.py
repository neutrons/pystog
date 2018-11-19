

class Nickel(object):

    # ---------------------------
    # Material info

    kwargs = {"rho": 0.0913841384754395,
              "<b_coh>^2": 106.09,
              "<b_tot^2>": 147.22}
    lammps_gr_filename = "nickel.gr"

    # ---------------------------
    # Real space

    real_space_filename = "nickel.real_space.dat"
    real_space_first = 45
    real_space_last = 53
    gofr_target = [0.036372,
                   0.832999,
                   5.705700,
                   12.894100,
                   10.489400,
                   3.267010,
                   0.416700,
                   0.021275]
    GofR_target = [-2.57284109375,
                   -0.45547376985,
                   13.1043856417,
                   33.8055086359,
                   27.5157162282,
                   6.703650364,
                   -1.75833641369,
                   -3.00652731657]
    GKofR_target = [-102.2312733,
                    -17.71713609,
                    499.227713,
                    1261.845069,
                    1006.730446,
                    240.5070909,
                    -61.882297,
                    -103.83293525]

    # ---------------------------
    # Reciprocal space

    reciprocal_space_filename = "nickel.reciprocal_space.dat"
    reciprocal_space_first = 150
    reciprocal_space_last = 157

    sq_target = [7.07469,
                 8.704824,
                 9.847706,
                 10.384142,
                 10.265869,
                 9.519633,
                 8.240809]
    fq_target = [18.345563,
                 23.422666,
                 27.07398,
                 28.903156,
                 28.724193,
                 26.581256,
                 22.73614]
    fq_keen_target = [644.463844,
                      817.404819,
                      938.653124,
                      995.563576,
                      983.016011,
                      903.847917,
                      768.177419]
    dcs_target = [791.683844,
                  964.624819,
                  1085.873124,
                  1142.783576,
                  1130.236011,
                  1051.067917,
                  915.397419]


class Argon(object):

    # ---------------------------
    # Material info

    kwargs = {"rho": 0.02138, "<b_coh>^2": 3.644, "<b_tot^2>": 5.435}
    lammps_gr_filename = "argon.gr"

    # ---------------------------
    # Real space

    real_space_filename = "argon.real_space.dat"
    real_space_first = 69
    real_space_last = 76

    gofr_target = [2.3774,
                   2.70072,
                   2.90777,
                   3.01835,
                   2.99808,
                   2.89997,
                   2.75178]
    GofR_target = [1.304478,
                   1.633527,
                   1.858025,
                   1.992835,
                   1.999663,
                   1.926998,
                   1.800236]
    GKofR_target = [5.019246,
                    6.197424,
                    6.951914,
                    7.354867,
                    7.281004,
                    6.923491,
                    6.383486]

    # ---------------------------
    # Reciprocal space

    reciprocal_space_filename = "argon.reciprocal_space.dat"
    reciprocal_space_first = 96
    reciprocal_space_last = 103

    sq_target = [2.59173,
                 2.706695,
                 2.768409,
                 2.770228,
                 2.71334,
                 2.605211,
                 2.458852]
    fq_target = [3.087955,
                 3.345121,
                 3.50145,
                 3.540457,
                 3.460946,
                 3.27463,
                 3.005236]
    fq_keen_target = [5.800262,
                      6.219195,
                      6.444083,
                      6.450712,
                      6.24341,
                      5.849389,
                      5.316058]
    dcs_target = [11.235262,
                  11.654195,
                  11.879083,
                  11.885712,
                  11.67841,
                  11.284389,
                  10.751058]
