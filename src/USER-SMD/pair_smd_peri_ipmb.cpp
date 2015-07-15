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

/* ----------------------------------------------------------------------
 Contributing author: Mike Parks (SNL)
 ------------------------------------------------------------------------- */

#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "pair_smd_peri_ipmb.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_peri_neigh_gcg.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
#define FORMAT1 "%60s : %g\n"
#define FORMAT2 "\n.............................. %s \n"

#if defined(__clang__) || defined (__GNUC__)
# define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
# define ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif

/* ---------------------------------------------------------------------- */

PairPeriGCG::PairPeriGCG(LAMMPS *lmp) :
		Pair(lmp) {
	for (int i = 0; i < 6; i++)
		virial[i] = 0.0;
	no_virial_fdotr_compute = 1;

	ifix_peri = -1;

	bulkmodulus = NULL;
	smax = syield = NULL;
	G0 = NULL;
	alpha = NULL;
	rho0 = NULL;
	c0 = NULL;
	contact_forces = NULL;
	HAVE_CONTACT_FORCES_GLOBAL = false;
	cutoff_global = 0.0;

	nBroken = 0;

	first = true;
}

/* ---------------------------------------------------------------------- */

PairPeriGCG::~PairPeriGCG() {
	if (ifix_peri >= 0)
		modify->delete_fix("PERI_NEIGH_GCG");

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(bulkmodulus);
		memory->destroy(smax);
		memory->destroy(syield);
		memory->destroy(G0);
		memory->destroy(alpha);
		memory->destroy(rho0);
		memory->destroy(c0);
		memory->destroy(contact_forces);
		memory->destroy(cutsq);
	}
}

/* ---------------------------------------------------------------------- */

void PairPeriGCG::compute(int eflag, int vflag) {
	int i, j, ii, jj, inum, jnum, itype, jtype;
	double xtmp, ytmp, ztmp, delx, dely, delz;
	double rsq, r, dr, evdwl, fpair;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double delta, stretch;
	double vxtmp, vytmp, vztmp, delvelx, delvely, delvelz, rcut, strain_rate;
	double c;
	double r0cut, delx0, dely0, delz0;
	double r_geom, radius_factor, dforce_dr, k, dt_crit;
	// ---------------------------------------------------------------------------------
	double **f = atom->f;
	double **x = atom->x;
	double **x0 = atom->x0;
	int *type = atom->type;
	double *e = atom->e;
	double **v = atom->v;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double *radiusSR = atom->contact_radius;
	double *rmass = atom->rmass;
	int *molecule = atom->molecule;
	int nlocal = atom->nlocal;
	int newton_pair = force->newton_pair;
	double **r0 = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->r0;
	double **plastic_stretch = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->plastic_stretch;
	tagint **partner = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->partner;
	int *npartner = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->npartner;
	double *vinter = ((FixPeriNeighGCG *) modify->fix[ifix_peri])->vinter;

	evdwl = 0.0;
	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (first) { // return on first call, because reference connectivity lists still needs to be built. Also zero quantities which are otherwise undefined.
		first = false;
		return;
	}

	stable_time_increment = 1.0e22;

	/* ----------------------- PERIDYNAMIC SHORT RANGE FORCES --------------------- */

	// neighbor list variables -- note that we use a granular neighbor list
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	if (HAVE_CONTACT_FORCES_GLOBAL) {

		for (ii = 0; ii < inum; ii++) {
			i = ilist[ii];
			xtmp = x[i][0];
			ytmp = x[i][1];
			ztmp = x[i][2];
			vxtmp = v[i][0];
			vytmp = v[i][1];
			vztmp = v[i][2];
			itype = type[i];
			jlist = firstneigh[i];
			jnum = numneigh[i];

			for (jj = 0; jj < jnum; jj++) {
				j = jlist[jj];
				j &= NEIGHMASK;

				jtype = type[j];

				if (!contact_forces[itype][jtype]) {
					continue;
				}

				delx = xtmp - x[j][0];
				dely = ytmp - x[j][1];
				delz = ztmp - x[j][2];
				rsq = delx * delx + dely * dely + delz * delz;

				/* We only let this pair of particles interact via short-range if their peridynamic bond is broken.
				 * The reduced cutoff radius below ensures this.
				 * If particles were not bonded initially (molecule[i] != molecule[j]), no reduction is performed
				 */
				if (molecule[i] == molecule[j]) {
					radius_factor = 1.0 - smax[itype][jtype];
				} else {
					radius_factor = 1.0;
				}

				// initial distance in reference config
				delx0 = x0[j][0] - x0[i][0];
				dely0 = x0[j][1] - x0[i][1];
				delz0 = x0[j][2] - x0[i][2];
				r0cut = sqrt(delx0 * delx0 + dely0 * dely0 + delz0 * delz0);

				rcut = radius_factor * (radiusSR[i] + radiusSR[j]);
				rcut = MIN(rcut, r0cut);

				if (rsq < rcut * rcut) {

					// Hertzian short-range forces
					r = sqrt(rsq);
					delta = rcut - r; // overlap distance
					r_geom = radius_factor * radius_factor * radiusSR[i] * radiusSR[j] / rcut;

					//assuming poisson ratio = 1/4 for 3d
					fpair = 1.066666667e0 * bulkmodulus[itype][jtype] * delta * sqrt(delta * r_geom);
					evdwl = fpair * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy

					// compute stable time increment
					dforce_dr = 1.5 * fpair / delta; // derivative of force w.r.t r
					k = 2.0 * dforce_dr / (rmass[i] + rmass[j]);
					dt_crit = 4.0 / sqrt(k);
					//printf("dt_crit is %f, df_dr=%f\n", dt_crit, dforce_dr);
					stable_time_increment = MIN(stable_time_increment, dt_crit);

					if (evflag) {
						ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
					}

					if (r > 0.0) {
						fpair = fpair / r; // divide by r so we can multiple with the non-normalized distance vector
					} else {
						fpair = 0.0;
					}

					f[i][0] += delx * fpair;
					f[i][1] += dely * fpair;
					f[i][2] += delz * fpair;

					if (j < nlocal) {
						f[j][0] -= delx * fpair;
						f[j][1] -= dely * fpair;
						f[j][2] -= delz * fpair;
					}

				}
			}
		}
	}

	/* ----------------------- PERIDYNAMIC BOND FORCES --------------------- */
	stable_time_increment = 9999999999.0;

	for (i = 0; i < nlocal; i++) {

		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jnum = npartner[i];

		for (jj = 0; jj < jnum; jj++) {

			if (partner[i][jj] == 0)
				continue;
			j = atom->map(partner[i][jj]);

			// check if lost a partner without first breaking bond
			if (j < 0) {
				partner[i][jj] = 0;
				continue;
			}

			if (molecule[i] != molecule[j]) {
				printf("ERROR: molecule[i] != molecule[j] :: itype=%d, imol=%d, jtype=%d, jmol=%d\n\n", itype, molecule[i], jtype,
						molecule[j]);
				error->all(FLERR, "molecule[i] != molecule[j]");
			}

			delx = xtmp - x[j][0];
			dely = ytmp - x[j][1];
			delz = ztmp - x[j][2];

			rsq = delx * delx + dely * dely + delz * delz;
			jtype = type[j];
			delta = radius[i] + radius[j];
			r = sqrt(rsq);
			dr = r - r0[i][jj];

			// avoid roundoff errors
			if (fabs(dr) < 2.2204e-016)
				dr = 0.0;

			// bond stretch
			stretch = dr / r0[i][jj]; // total stretch

			// subtract plastic stretch from current stretch
			stretch -= plastic_stretch[i][jj];

			// alternative plasticity based on plastic stretch
			if (stretch > syield[itype][jtype]) {
				double plastic_stretch_increment = stretch - syield[itype][jtype];
				plastic_stretch[i][jj] += plastic_stretch_increment;
				stretch = syield[itype][jtype];
			}

			if (domain->dimension == 2) {
				c = 4.5 * bulkmodulus[itype][jtype] * (1.0 / vinter[i] + 1.0 / vinter[j]);
			} else {
				c = 9.0 * bulkmodulus[itype][jtype] * (1.0 / vinter[i] + 1.0 / vinter[j]);
			}

			// force computation -- note we divide by a factor of r
			evdwl = 0.5 * c * stretch * stretch * vfrac[i] * vfrac[j];
			fpair = -c * vfrac[i] * vfrac[j] * stretch / r0[i][jj];

			dforce_dr = c * vfrac[i] * vfrac[j] / r0[i][jj];
			k = 2.0 * dforce_dr / (rmass[i] + rmass[j]);
			dt_crit = 4.0 / sqrt(k);
			if (k < 0.0) {
				error->one(FLERR, "k is negative");
			}
			//printf("dt_crit is %f, df_dr=%f\n", dt_crit, dforce_dr);
			stable_time_increment = MIN(stable_time_increment, dt_crit);

			// artificial viscosity
			if (alpha[itype][jtype] > 0.0) {
				delvelx = vxtmp - v[j][0];
				delvely = vytmp - v[j][1];
				delvelz = vztmp - v[j][2];
				strain_rate = (delx * delvelx + dely * delvely + delz * delvelz) / (r * r0[i][jj]); // pair-wise strain rate
				fpair -= rho0[itype][jtype] * alpha[itype][jtype] * c0[itype][jtype] * strain_rate;
			}

			// project force -- missing factor of r is recovered here as delx, dely ... are not unit vectors
			if (r > 0.0)
				fpair = fpair / r;
			else
				fpair = 0.0;
			f[i][0] += delx * fpair;
			f[i][1] += dely * fpair;
			f[i][2] += delz * fpair;

			if (evflag) {
				// since I-J is double counted, set newton off & use 1/2 factor and I,I
				ev_tally(i, i, nlocal, 0, 0.5 * evdwl, 0.0, 0.5 * fpair, delx, dely, delz);
				//printf("broken evdwl=%f, norm = %f %f\n", evdwl, vinter[i], vinter[j]);
			}

			if ((r - r0[i][jj]) / r0[i][jj] > smax[itype][jtype]) {
				partner[i][jj] = 0;
				nBroken += 1;
				e[i] += 0.5 * evdwl;
				//printf("breaking bond\n");
			}

		}

	}

	//printf("dtmin in pair is is %f, nlocal is %d\n", stable_time_increment, nlocal);

}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairPeriGCG::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(rho0, n + 1, n + 1, "pair:rho0");
	memory->create(c0, n + 1, n + 1, "pair:c0");
	memory->create(bulkmodulus, n + 1, n + 1, "pair:kspring");
	memory->create(smax, n + 1, n + 1, "pair:smax");
	memory->create(syield, n + 1, n + 1, "pair:syield");
	memory->create(G0, n + 1, n + 1, "pair:G0");
	memory->create(alpha, n + 1, n + 1, "pair:alpha");
	memory->create(contact_forces, n + 1, n + 1, "pair:contact_forces");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairPeriGCG::settings(int narg, char **arg) {
	if (narg != 0)
		error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairPeriGCG::coeff(int narg, char **arg) {
	if (narg != 9)
		error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	double rho0_one = atof(arg[2]);
	double bulkmodulus_one = atof(arg[3]);
	double smax_one = atof(arg[4]);
	double G0_one = atof(arg[5]);
	double alpha_one = atof(arg[6]);
	double syield_one = atof(arg[7]);
	double c0_one = sqrt(bulkmodulus_one / rho0_one);
	bool contact_forces_one = force->inumeric(FLERR, arg[8]) != 0;
	if (contact_forces_one)
		HAVE_CONTACT_FORCES_GLOBAL = true;

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {

		for (int j = MAX(jlo, i); j <= jhi; j++) {
			rho0[i][j] = rho0_one;
			bulkmodulus[i][j] = bulkmodulus_one;
			smax[i][j] = smax_one;
			syield[i][j] = syield_one;
			G0[i][j] = G0_one;
			alpha[i][j] = alpha_one;
			contact_forces[i][j] = contact_forces_one;
			c0[i][j] = c0_one;
			setflag[i][j] = 1;
			count++;
		}

		if (comm->me == 0) {
			printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
			printf("...BOND BASED PERIDYNAMIC PROPERTIES OF PARTICLE TYPE %d\n\n", i);
			printf(FORMAT1, "reference density", rho0[i][i]);
			printf(FORMAT1, "reference speed of sound", c0[i][i]);
			printf(FORMAT1, "bulk modulus", bulkmodulus[i][i]);
			printf(FORMAT1, "yield stretch", syield[i][i]);
			printf(FORMAT1, "failure stretch", smax[i][i]);
			printf(FORMAT1, "energy release rate (negative if deactivated)", G0[i][i]);
			printf(FORMAT1, "linear artificial viscosity coefficient", alpha[i][i]);
			printf("%60s : %s\n", "self contact forces", contact_forces[i][i] ? "true" : "false");
			printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
		}
	}

	if (count == 0)
		error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairPeriGCG::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	bulkmodulus[j][i] = bulkmodulus[i][j];
	alpha[j][i] = alpha[i][j];
	smax[j][i] = smax[i][j];
	syield[j][i] = syield[i][j];
	G0[j][i] = G0[i][j];
	contact_forces[j][i] = contact_forces[i][j];

	return cutoff_global;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */
ATTRIBUTE_NO_SANITIZE_ADDRESS
void PairPeriGCG::init_style() {
	int i;

// error checks

	if (!atom->x0_flag)
		error->all(FLERR, "Pair style peri requires atom style with x0");
	if (atom->map_style == 0)
		error->all(FLERR, "Pair peri requires an atom map, see atom_modify");

// if first init, create Fix needed for storing fixed neighbors

	if (ifix_peri == -1) {
		char **fixarg = new char*[3];
		fixarg[0] = (char *) "PERI_NEIGH_GCG";
		fixarg[1] = (char *) "all";
		fixarg[2] = (char *) "PERI_NEIGH_GCG";
		modify->add_fix(3, fixarg);
		delete[] fixarg;
	}

// find associated PERI_NEIGH fix that must exist
// could have changed locations in fix list since created

	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "PERI_NEIGH_GCG") == 0)
			ifix_peri = i;
	if (ifix_peri == -1)
		error->all(FLERR, "Fix peri neigh GCG does not exist");

// request a granular neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;

	fix_peri_neigh_gcg = (FixPeriNeighGCG *) modify->fix[modify->nfix - 1];
	fix_peri_neigh_gcg->pair = this;

	double *radius = atom->radius;
	int nlocal = atom->nlocal;
	double maxrad_one = 0.0;

	for (i = 0; i < nlocal; i++) {
		maxrad_one = MAX(maxrad_one, 2.0 * radius[i]);
	}

	//printf("proc %d has maxrad %f\n", comm->me, maxrad_one);

	MPI_Allreduce(&maxrad_one, &cutoff_global, atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
	cutoff_global = maxrad_one;

	if (comm->me == 0) {
		printf("global cutoff for pair style peri/gcg is %f\n", cutoff_global);
	}
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairPeriGCG::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void PairPeriGCG::write_restart(FILE *fp) {
	int i, j;
	for (i = 1; i <= atom->ntypes; i++)
		for (j = i; j <= atom->ntypes; j++) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&bulkmodulus[i][j], sizeof(double), 1, fp);
				fwrite(&smax[i][j], sizeof(double), 1, fp);
				fwrite(&syield[i][j], sizeof(double), 1, fp);
				fwrite(&alpha[i][j], sizeof(double), 1, fp);
			}
		}
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void PairPeriGCG::read_restart(FILE *fp) {
	allocate();

	int i, j;
	int me = comm->me;
	for (i = 1; i <= atom->ntypes; i++)
		for (j = i; j <= atom->ntypes; j++) {
			if (me == 0)
				fread(&setflag[i][j], sizeof(int), 1, fp);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (me == 0) {
					fread(&bulkmodulus[i][j], sizeof(double), 1, fp);
					fread(&smax[i][j], sizeof(double), 1, fp);
					fread(&syield[i][j], sizeof(double), 1, fp);
					fread(&alpha[i][j], sizeof(double), 1, fp);
				}
				MPI_Bcast(&bulkmodulus[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&smax[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&syield[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&alpha[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairPeriGCG::memory_usage() {

	return 0.0;
}

/* ----------------------------------------------------------------------
 extract method to access stuff computed within this pair style
 ------------------------------------------------------------------------- */

void *PairPeriGCG::extract(const char *str, int &i) {
	//printf("in PairPeriGCG::extract\n");
	if (strcmp(str, "smd/peri_ipmb/stable_time_increment_ptr") == 0) {
		return (void *) &stable_time_increment;
	}

	return NULL;

}
