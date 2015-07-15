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
 Contributing authors: Mike Parks (SNL), Ezwanur Rahman, J.T. Foster (UTSA)
 ------------------------------------------------------------------------- */

#include "math.h"
#include "fix_peri_neigh_gcg.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "update.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "lattice.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;
#define DELTA 16384

/* ---------------------------------------------------------------------- */

FixPeriNeighGCG::FixPeriNeighGCG(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	restart_global = 1;
	restart_peratom = 1;
	first = 1;

	// perform initial allocation of atom-based arrays
	// register with atom class
	// set maxpartner = 1 as placeholder

	maxpartner = 1;
	npartner = NULL;
	partner = NULL;
	r0 = plastic_stretch = NULL;
	vinter = NULL;

	grow_arrays(atom->nmax);
	atom->add_callback(0);
	atom->add_callback(1);
	atom->add_callback(2); // for border communication

	// initialize npartner to 0 so atom migration is OK the 1st time

	int nlocal = atom->nlocal;
	for (int i = 0; i < nlocal; i++)
		npartner[i] = 0;

	// set comm sizes needed by this fix

	/*
	 * need both pack_forward_comm / unpack_forward_comm and
	 * pack_border / unpack_border routines for communicating Shepard volume sum to ghosts
	 */
	comm_forward = 1;
	comm_border = 1;
}

/* ---------------------------------------------------------------------- */

FixPeriNeighGCG::~FixPeriNeighGCG() {
	// unregister this fix so atom class doesn't invoke it any more

	atom->delete_callback(id, 0);
	atom->delete_callback(id, 1);

	// delete locally stored arrays

	memory->destroy(npartner);
	memory->destroy(partner);
	memory->destroy(r0);
	memory->destroy(plastic_stretch);
	memory->destroy(vinter);
}

/* ---------------------------------------------------------------------- */

int FixPeriNeighGCG::setmask() {
	int mask = 0;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixPeriNeighGCG::init() {
	if (!first)
		return;

	if (atom->tag_enable == 0)
		error->all(FLERR, "Pair style smd/peri_ipmb requires atoms have IDs");
}

/* ----------------------------------------------------------------------
 For minimization: setup as with dynamics
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::min_setup(int vflag) {
	setup(vflag);
}

/* ----------------------------------------------------------------------
 create initial list of neighbor partners via call to neighbor->build()
 must be done in setup (not init) since fix init comes before neigh init
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::setup(int vflag) {
	int i, j, ii, jj, itype, jtype, inum, jnum;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq, cutsq;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;

	double **x = atom->x;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	int *type = atom->type;
	tagint *tag = atom->tag;
	tagint *molecule = atom->molecule;
	int nlocal = atom->nlocal;

	NeighList *list = pair->list;
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	// only build list of bonds on very first run

	if (!first)
		return;
	first = 0;

	for (i = 0; i < nlocal; i++) { // make sure all atoms on this proc have initially zero partners.
		// this avoids the problem that the loop below does not include all atoms on this proc, e.g. if
		// a hybrid/overlay pair style is used.
		npartner[i] = 0;
	}

	// scan neighbor list to set maxpartner

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];

		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (molecule[i] == molecule[j]) {

				delx = xtmp - x[j][0];
				dely = ytmp - x[j][1];
				delz = ztmp - x[j][2];
				rsq = delx * delx + dely * dely + delz * delz;
				jtype = type[j];

				cutsq = (radius[i] + radius[j]) * (radius[i] + radius[j]);

				if (rsq <= cutsq) {
					npartner[i]++;

					if (j < nlocal) {
						npartner[j]++;
					}
				} // end check for distance
			} // end check for molecule[i] == molecule[j]
		}
	}

	maxpartner = 0;
	for (i = 0; i < nlocal; i++)
		maxpartner = MAX(maxpartner, npartner[i]);
	int maxall;
	MPI_Allreduce(&maxpartner, &maxall, 1, MPI_INT, MPI_MAX, world);
	maxpartner = maxall;

	// realloc arrays with correct value for maxpartner

	memory->destroy(partner);
	memory->destroy(r0);
	memory->destroy(plastic_stretch);
	memory->destroy(npartner);

	npartner = NULL;
	partner = NULL;
	r0 = plastic_stretch = NULL;
	nmax = atom->nmax;
	grow_arrays(nmax);

	// create partner list and r0 values from neighbor list
	// compute vinter for each atom

	for (i = 0; i < nmax; i++) {
		npartner[i] = 0;
		vinter[i] = 0.0;
	}

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			if (molecule[i] == molecule[j]) {

				delx = xtmp - x[j][0];
				dely = ytmp - x[j][1];
				delz = ztmp - x[j][2];
				rsq = delx * delx + dely * dely + delz * delz;
				jtype = type[j];

				cutsq = (radius[i] + radius[j]) * (radius[i] + radius[j]);

				if (rsq <= cutsq) {
					partner[i][npartner[i]] = tag[j];
					r0[i][npartner[i]] = sqrt(rsq);
					plastic_stretch[i][npartner[i]] = 0.0;
					npartner[i]++;
					vinter[i] += vfrac[j];

					if (j < nlocal) {
						partner[j][npartner[j]] = tag[i];
						r0[j][npartner[j]] = sqrt(rsq);
						plastic_stretch[j][npartner[j]] = 0.0;
						npartner[j]++;
						vinter[j] += vfrac[i];
					}
				} // end check for distance
			} // end check for molecule[i] == molecule[j]
		}
	}

	// sanity check: does any atom appear twice in any neigborlist?
	// should only be possible if using pbc and domain < 2*delta

	if (domain->xperiodic || domain->yperiodic || domain->zperiodic) {
		for (i = 0; i < nlocal; i++) {
			jnum = npartner[i];
			//printf("\n\n partners of atom tag %d:\n", tag[i]);
			for (jj = 0; jj < jnum; jj++) {
				//printf("%d\n", partner[i][jj]);
				for (int kk = jj + 1; kk < jnum; kk++) {
					if (partner[i][jj] == partner[i][kk])
						error->one(FLERR, "Duplicate particle in PeriDynamic bond - "
								"simulation box is too small");
				}
			}
		}
	}

	// communicate vinter to ghosts, needed in pair

	comm->forward_comm_fix(this);

	// bond statistics

	int n = 0;
	for (i = 0; i < nlocal; i++)
		n += npartner[i];
	int nall;
	MPI_Allreduce(&n, &nall, 1, MPI_INT, MPI_SUM, world);

	if (comm->me == 0) {
		if (screen) {
			fprintf(screen, "Peridynamic bonds:\n");
			fprintf(screen, "  total # of bonds = %d\n", nall);
			fprintf(screen, "  bonds/atom = %g\n", (double) nall / atom->natoms);
		}
		if (logfile) {
			fprintf(logfile, "Peridynamic bonds:\n");
			fprintf(logfile, "  total # of bonds = %d\n", nall);
			fprintf(logfile, "  bonds/atom = %g\n", (double) nall / atom->natoms);
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double FixPeriNeighGCG::memory_usage() {
	int nmax = atom->nmax;
	int bytes = nmax * sizeof(int);
	bytes += nmax * maxpartner * sizeof(tagint);
	bytes += nmax * maxpartner * sizeof(double);
	bytes += nmax * sizeof(double);
	bytes += nmax * sizeof(double);
	return bytes;
}

/* ----------------------------------------------------------------------
 allocate local atom-based arrays
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::grow_arrays(int nmax) {
	//printf("in FixPeriNeighGCG::grow_arrays, nmax=%d, maxpartner=%d\n", nmax, maxpartner);
	memory->grow(npartner, nmax, "peri_neigh:npartner");
	memory->grow(partner, nmax, maxpartner, "peri_neigh:partner");
	memory->grow(r0, nmax, maxpartner, "peri_neigh:r0");
	memory->grow(plastic_stretch, nmax, maxpartner, "peri_neigh:r1");
	memory->grow(vinter, nmax, "peri_neigh:vinter");
}

/* ----------------------------------------------------------------------
 copy values within local atom-based arrays
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::copy_arrays(int i, int j, int delflag) {
	npartner[j] = npartner[i];
	for (int m = 0; m < npartner[j]; m++) {
		partner[j][m] = partner[i][m];
		r0[j][m] = r0[i][m];
		plastic_stretch[j][m] = plastic_stretch[i][m];
	}
	vinter[j] = vinter[i];
}

/* ----------------------------------------------------------------------
 pack values for border communication at re-neighboring
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::pack_border(int n, int *list, double *buf) {
	int i, j;
	//printf("FixPeriNeighGCG::pack_border\n");

	int m = 0;

	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = vinter[j];
	}

	return m;
}

/* ----------------------------------------------------------------------
 unpack values for border communication at re-neighboring
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::unpack_border(int n, int first, double *buf) {
	int i, last;

	int m = 0;
	last = first + n;
	for (i = first; i < last; i++)
		vinter[i] = buf[m++];

	return m;
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for exchange with another proc
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::pack_exchange(int i, double *buf) {
	//printf("in FixPeriNeighGCG::pack_exchange ------------------------------------------\n");

	int m = 0;
	buf[m++] = npartner[i];
	for (int n = 0; n < npartner[i]; n++) {
		buf[m++] = partner[i][n];
		buf[m++] = r0[i][n];
		buf[m++] = plastic_stretch[i][n];
	}
	buf[m++] = vinter[i];
	return m;
}

/* ----------------------------------------------------------------------
 unpack values in local atom-based arrays from exchange with another proc
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::unpack_exchange(int nlocal, double *buf) {

	if (nlocal == nmax) {

		//printf("nlocal=%d, nmax=%d\n", nlocal, nmax);

		nmax = nmax / DELTA * DELTA;
		nmax += DELTA;
		grow_arrays(nmax);

		error->message(FLERR,
				"in FixPeriNeighGCG::unpack_exchange: local arrays too small for receiving partner information; growing arrays");
	}
	//printf("nlocal=%d, nmax=%d\n", nlocal, nmax);

	int m = 0;
	npartner[nlocal] = static_cast<int>(buf[m++]);
	for (int n = 0; n < npartner[nlocal]; n++) {
		partner[nlocal][n] = static_cast<tagint>(buf[m++]);
		r0[nlocal][n] = buf[m++];
		plastic_stretch[nlocal][n] = buf[m++];
	}
	vinter[nlocal] = buf[m++];
	return m;
}

/* ----------------------------------------------------------------------
 pack entire state of Fix into one write
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::write_restart(FILE *fp) {
	int n = 0;
	double list[2];
	list[n++] = first;
	list[n++] = maxpartner;

	if (comm->me == 0) {
		int size = n * sizeof(double);
		fwrite(&size, sizeof(int), 1, fp);
		fwrite(list, sizeof(double), n, fp);
	}
}

/* ----------------------------------------------------------------------
 use state info from restart file to restart the Fix
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::restart(char *buf) {
	int n = 0;
	double *list = (double *) buf;

	first = static_cast<int>(list[n++]);
	maxpartner = static_cast<int>(list[n++]);

	// grow 2D arrays now, cannot change size of 2nd array index later

	grow_arrays(atom->nmax);
}

/* ----------------------------------------------------------------------
 pack values in local atom-based arrays for restart file
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::pack_restart(int i, double *buf) {
	int m = 0;
	buf[m++] = 3 * npartner[i] + 3;
	buf[m++] = npartner[i];
	for (int n = 0; n < npartner[i]; n++) {
		buf[m++] = partner[i][n];
		buf[m++] = r0[i][n];
		buf[m++] = plastic_stretch[i][n];
	}
	buf[m++] = vinter[i];
	return m;
}

/* ----------------------------------------------------------------------
 unpack values from atom->extra array to restart the fix
 ------------------------------------------------------------------------- */

void FixPeriNeighGCG::unpack_restart(int nlocal, int nth) {

	double **extra = atom->extra;

	// skip to Nth set of extra values

	int m = 0;
	for (int i = 0; i < nth; i++)
		m += static_cast<int>(extra[nlocal][m]);
	m++;

	npartner[nlocal] = static_cast<int>(extra[nlocal][m++]);
	for (int n = 0; n < npartner[nlocal]; n++) {
		partner[nlocal][n] = static_cast<tagint>(extra[nlocal][m++]);
		r0[nlocal][n] = extra[nlocal][m++];
		plastic_stretch[nlocal][n] = extra[nlocal][m++];
	}
	vinter[nlocal] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
 maxsize of any atom's restart data
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::maxsize_restart() {
	return 2 * maxpartner + 3;
}

/* ----------------------------------------------------------------------
 size of atom nlocal's restart data
 ------------------------------------------------------------------------- */

int FixPeriNeighGCG::size_restart(int nlocal) {
	return 2 * npartner[nlocal] + 3;
}


/* ---------------------------------------------------------------------- */

int FixPeriNeighGCG::pack_forward_comm(int n, int *list, double *buf,
                                    int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = vinter[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixPeriNeighGCG::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    vinter[i] = buf[m++];
}
