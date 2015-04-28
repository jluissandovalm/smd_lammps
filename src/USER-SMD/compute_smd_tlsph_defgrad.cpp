/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "string.h"
#include "compute_smd_tlsph_defgrad.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <iostream>
#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include <Eigen/Eigen>
using namespace Eigen;
using namespace std;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHDefgrad::ComputeSMDTLSPHDefgrad(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute smd/tlsph_defgrad command");

    peratom_flag = 1;
    size_peratom_cols = 10;

    nmax = 0;
    defgradVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHDefgrad::~ComputeSMDTLSPHDefgrad() {
    memory->sfree(defgradVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHDefgrad::init() {

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "smd/tlsph_defgrad") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute smd/tlsph_defgrad");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHDefgrad::compute_peratom() {
    invoked_peratom = update->ntimestep;

    // grow vector array if necessary

    if (atom->nlocal > nmax) {
        memory->destroy(defgradVector);
        nmax = atom->nmax;
        memory->create(defgradVector, nmax, size_peratom_cols, "defgradVector");
        array_atom = defgradVector;
    }

    // copy data to output array
    int itmp = 0;
    double *detF = (double *) force->pair->extract("smd/tlsph/detF_ptr", itmp);
    if (detF == NULL) {
        error->all(FLERR, "compute smd/tlsph_defgrad failed to access detF array");
    }

    Matrix3d *Fincr = (Matrix3d *) force->pair->extract("smd/tlsph/Fincr_ptr", itmp);
    if (Fincr == NULL) {
        error->all(FLERR, "compute smd/tlsph_defgrad failed to access Fincr array");
    }

    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            defgradVector[i][0] = Fincr[i](0, 0);
            defgradVector[i][1] = Fincr[i](0, 1);
            defgradVector[i][2] = Fincr[i](0, 2);
            defgradVector[i][3] = Fincr[i](1, 0);
            defgradVector[i][4] = Fincr[i](1, 1);
            defgradVector[i][5] = Fincr[i](1, 2);
            defgradVector[i][6] = Fincr[i](2, 0);
            defgradVector[i][7] = Fincr[i](2, 1);
            defgradVector[i][8] = Fincr[i](2, 2);
            defgradVector[i][9] = detF[i];
        } else {
            for (int j = 0; j < size_peratom_cols; j++) {
                defgradVector[i][j] = 0.0;
            }
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHDefgrad::memory_usage() {
    double bytes = size_peratom_cols * nmax * sizeof(double);
    return bytes;
}
