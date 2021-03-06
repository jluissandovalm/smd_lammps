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

/* ----------------------------------------------------------------------
 Contributing author: Paul Crozier (SNL)
 ------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "fix_smd_heat.h"
#include "atom.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum {
	CONSTANT, EQUAL, ATOM
};

/* ---------------------------------------------------------------------- */

FixSMDHeat::FixSMDHeat(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg < 4)
		error->all(FLERR, "Illegal fix smd/heat command");

	scalar_flag = 1;
	global_freq = 1;
	extscalar = 0;

	min_e = 0.001;
	max_e = 0.01;

	nevery = force->inumeric(FLERR, arg[3]);
	if (nevery <= 0)
		error->all(FLERR, "Illegal fix smd/heat command");

	hstr = NULL;

	if (strstr(arg[4], "v_") == arg[4]) {
		int n = strlen(&arg[4][2]) + 1;
		hstr = new char[n];
		strcpy(hstr, &arg[4][2]);
	} else {
		heat_input = force->numeric(FLERR, arg[4]);
		hstyle = CONSTANT;
	}

	// optional args

	iregion = -1;
	idregion = NULL;

	int iarg = 5;
	while (iarg < narg) {
		if (strcmp(arg[iarg], "region") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix smd/heat command");
			iregion = domain->find_region(arg[iarg + 1]);
			if (iregion == -1)
				error->all(FLERR, "Region ID for fix smd/heat does not exist");
			int n = strlen(arg[iarg + 1]) + 1;
			idregion = new char[n];
			strcpy(idregion, arg[iarg + 1]);
			iarg += 2;
		} else
			error->all(FLERR, "Illegal fix smd/heat command");
	}

	maxatom = 0;
}

/* ---------------------------------------------------------------------- */

FixSMDHeat::~FixSMDHeat() {
	delete[] hstr;
	delete[] idregion;
}

/* ---------------------------------------------------------------------- */

int FixSMDHeat::setmask() {
	int mask = 0;
	mask |= END_OF_STEP;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDHeat::init() {
	// set index and check validity of region

	if (iregion >= 0) {
		iregion = domain->find_region(idregion);
		if (iregion == -1)
			error->all(FLERR, "Region ID for fix smd/heat does not exist");
	}

	// check variable

	if (hstr) {
		hvar = input->variable->find(hstr);
		if (hvar < 0)
			error->all(FLERR, "Variable name for fix smd/heat does not exist");
		if (input->variable->equalstyle(hvar))
			hstyle = EQUAL;
		else if (input->variable->atomstyle(hvar))
			hstyle = ATOM;
		else
			error->all(FLERR, "Variable for fix smd/heat is invalid style");
	}

	// cannot have 0 atoms in group

	if (group->count(igroup) == 0)
		error->all(FLERR, "Fix heat group has no atoms");
}

/* ---------------------------------------------------------------------- */

void FixSMDHeat::end_of_step() {
	int i;
	double heat;

	double **x = atom->x;
	double *e = atom->e;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	double *rmass = atom->rmass;
	// evaluate variable

	if (hstyle != CONSTANT) {
		modify->clearstep_compute();
		if (hstyle == EQUAL)
			heat_input = input->variable->compute_equal(hvar);
		modify->addstep_compute(update->ntimestep + nevery);
	}

	Region *region = NULL;
	if (iregion >= 0) {
		region = domain->regions[iregion];
		region->prematch();
	}

	heat = heat_input * nevery * update->dt; // input: heat_input [energy / (mass * time)]

	if (hstyle != ATOM) {
		if (iregion < 0) {
			for (i = 0; i < nlocal; i++)
				if (mask[i] & groupbit) {
					e[i] += heat * rmass[i];
					e[i] = MAX(e[i], min_e);
					e[i] = MIN(e[i], max_e);
				}
		} else {
			for (int i = 0; i < nlocal; i++)
				if (mask[i] & groupbit && region->match(x[i][0], x[i][1], x[i][2])) {
					e[i] += heat * rmass[i];
					e[i] = MAX(e[i], min_e);
					e[i] = MIN(e[i], max_e);
				}
		}
	} else {
		if (iregion < 0) {
			for (i = 0; i < nlocal; i++) {
				if (mask[i] & groupbit) {
					e[i] += heat * rmass[i];
					e[i] = MAX(e[i], min_e);
					e[i] = MIN(e[i], max_e);
				}
			}
		} else {
			for (i = 0; i < nlocal; i++) {
				if (mask[i] & groupbit && region->match(x[i][0], x[i][1], x[i][2])) {
					e[i] += heat * rmass[i];
					e[i] = MAX(e[i], min_e);
					e[i] = MIN(e[i], max_e);
				}
			}
		}
	}
}

/* ---------------------------------------------------------------------- */

double FixSMDHeat::compute_scalar() {
	double scale_sum = 0.0;
	double scale_sum_all = 0.0;
	MPI_Allreduce(&scale_sum, &scale_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
	return scale_sum_all;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double FixSMDHeat::memory_usage() {
	double bytes = 0.0;
	return bytes;
}
