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

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "create_smd_tri_boundary.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "comm.h"
#include "irregular.h"
#include "modify.h"
#include "force.h"
#include "special.h"
#include "fix.h"
#include "compute.h"
#include "domain.h"
#include "lattice.h"
#include "region.h"
#include "input.h"
#include "variable.h"
#include "random_park.h"
#include "random_mars.h"
#include "math_extra.h"
#include "math_const.h"
#include "error.h"
#include "memory.h"
#include <Eigen/Eigen>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace MathConst;

#define BIG 1.0e30
#define EPSILON 1.0e-6

enum {
	BOX, REGION, SINGLE, RANDOM
};
enum {
	ATOM, MOLECULE
};
enum {
	LAYOUT_UNIFORM, LAYOUT_NONUNIFORM, LAYOUT_TILED
};
// several files

/* ---------------------------------------------------------------------- */

CreateSmdTriBoundary::CreateSmdTriBoundary(LAMMPS *lmp) :
		Pointers(lmp) {
}

/* ---------------------------------------------------------------------- */

void CreateSmdTriBoundary::command(int narg, char **arg) {
	if (domain->box_exist == 0)
		error->all(FLERR, "Create_atoms command before simulation box is defined");
	if (modify->nfix_restart_peratom)
		error->all(FLERR, "Cannot create_atoms after "
				"reading restart file with per-atom info");

	// parse arguments

	if (narg != 3)
		error->all(FLERR, "Illegal create_atoms command");

	filename.assign(arg[0]);
	wall_particle_type = force->inumeric(FLERR, arg[1]);
	wall_molecule_id = force->inumeric(FLERR, arg[2]);
	if (wall_molecule_id < 65535) {
		error->one(FLERR, "wall molcule id must be >= 65535\n");
	}

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("reading trianglulated surface from file: %s\n", filename.c_str());
		printf("... triangulated surface has particle type %d \n", wall_particle_type);
		printf("... triangulated surface has molecule id %d \n", wall_molecule_id);

	}

	// error checks

	if ((wall_particle_type <= 0) || (wall_particle_type > atom->ntypes))
		error->all(FLERR, "Invalid atom type in create_atoms command");

	// set bounds for my proc in sublo[3] & subhi[3]
	// if periodic and style = BOX or REGION, i.e. using lattice:
	//   should create exactly 1 atom when 2 images are both "on" the boundary
	//   either image may be slightly inside/outside true box due to round-off
	//   if I am lo proc, decrement lower bound by EPSILON
	//     this will insure lo image is created
	//   if I am hi proc, decrement upper bound by 2.0*EPSILON
	//     this will insure hi image is not created
	//   thus insertion box is EPSILON smaller than true box
	//     and is shifted away from true boundary
	//     which is where atoms are likely to be generated

	triclinic = domain->triclinic;

	double epsilon[3];
	if (triclinic)
		epsilon[0] = epsilon[1] = epsilon[2] = EPSILON;
	else {
		epsilon[0] = domain->prd[0] * EPSILON;
		epsilon[1] = domain->prd[1] * EPSILON;
		epsilon[2] = domain->prd[2] * EPSILON;
	}

	if (triclinic == 0) {
		sublo[0] = domain->sublo[0];
		subhi[0] = domain->subhi[0];
		sublo[1] = domain->sublo[1];
		subhi[1] = domain->subhi[1];
		sublo[2] = domain->sublo[2];
		subhi[2] = domain->subhi[2];
	} else {
		sublo[0] = domain->sublo_lamda[0];
		subhi[0] = domain->subhi_lamda[0];
		sublo[1] = domain->sublo_lamda[1];
		subhi[1] = domain->subhi_lamda[1];
		sublo[2] = domain->sublo_lamda[2];
		subhi[2] = domain->subhi_lamda[2];
	}

	if (comm->layout != LAYOUT_TILED) {
		if (domain->xperiodic) {
			if (comm->myloc[0] == 0)
				sublo[0] -= epsilon[0];
			if (comm->myloc[0] == comm->procgrid[0] - 1)
				subhi[0] -= 2.0 * epsilon[0];
		}
		if (domain->yperiodic) {
			if (comm->myloc[1] == 0)
				sublo[1] -= epsilon[1];
			if (comm->myloc[1] == comm->procgrid[1] - 1)
				subhi[1] -= 2.0 * epsilon[1];
		}
		if (domain->zperiodic) {
			if (comm->myloc[2] == 0)
				sublo[2] -= epsilon[2];
			if (comm->myloc[2] == comm->procgrid[2] - 1)
				subhi[2] -= 2.0 * epsilon[2];
		}
	} else {
		if (domain->xperiodic) {
			if (comm->mysplit[0][0] == 0.0)
				sublo[0] -= epsilon[0];
			if (comm->mysplit[0][1] == 1.0)
				subhi[0] -= 2.0 * epsilon[0];
		}
		if (domain->yperiodic) {
			if (comm->mysplit[1][0] == 0.0)
				sublo[1] -= epsilon[1];
			if (comm->mysplit[1][1] == 1.0)
				subhi[1] -= 2.0 * epsilon[1];
		}
		if (domain->zperiodic) {
			if (comm->mysplit[2][0] == 0.0)
				sublo[2] -= epsilon[2];
			if (comm->mysplit[2][1] == 1.0)
				subhi[2] -= 2.0 * epsilon[2];
		}
	}

	// read triangles and create particles

	bigint natoms_previous = atom->natoms;
	int nlocal_previous = atom->nlocal;

	read_triangles(0);

	// invoke set_arrays() for fixes/computes/variables
	//   that need initialization of attributes of new atoms
	// don't use modify->create_attributes() since would be inefficient
	//   for large number of atoms
	// note that for typical early use of create_atoms,
	//   no fixes/computes/variables exist yet

	int nlocal = atom->nlocal;
	for (int m = 0; m < modify->nfix; m++) {
		Fix *fix = modify->fix[m];
		if (fix->create_attribute)
			for (int i = nlocal_previous; i < nlocal; i++)
				fix->set_arrays(i);
	}
	for (int m = 0; m < modify->ncompute; m++) {
		Compute *compute = modify->compute[m];
		if (compute->create_attribute)
			for (int i = nlocal_previous; i < nlocal; i++)
				compute->set_arrays(i);
	}
	for (int i = nlocal_previous; i < nlocal; i++)
		input->variable->set_arrays(i);

	// set new total # of atoms and error check

	bigint nblocal = atom->nlocal;
	MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);
	if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
		error->all(FLERR, "Too many total atoms");

	// add IDs for newly created atoms
	// check that atom IDs are valid

	if (atom->tag_enable)
		atom->tag_extend();
	atom->tag_check();

	// create global mapping of atoms
	// zero nghost in case are adding new atoms to existing atoms

	if (atom->map_style) {
		atom->nghost = 0;
		atom->map_init();
		atom->map_set();
	}

	// print status
	if (comm->me == 0) {
		if (screen) {
			printf("... finished reading triangulated surface\n");
			fprintf(screen, "... created " BIGINT_FORMAT " atoms\n", atom->natoms - natoms_previous);
			printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
		}
		if (logfile) {
			fprintf(logfile, "... finished reading triangulated surface\n");
			fprintf(logfile, "... created " BIGINT_FORMAT " atoms\n", atom->natoms - natoms_previous);
			fprintf(logfile, ">>========>>========>>========>>========>>========>>========>>========>>========\n");
		}
	}

}

/* ----------------------------------------------------------------------
 function to determine number of values in a text line
 ------------------------------------------------------------------------- */

int CreateSmdTriBoundary::count_words(const char *line) {
	int n = strlen(line) + 1;
	char *copy;
	memory->create(copy, n, "atom:copy");
	strcpy(copy, line);

	char *ptr;
	if ((ptr = strchr(copy, '#')))
		*ptr = '\0';

	if (strtok(copy, " \t\n\r\f") == NULL) {
		memory->destroy(copy);
		return 0;
	}
	n = 1;
	while (strtok(NULL, " \t\n\r\f"))
		n++;

	memory->destroy(copy);
	return n;
}

/* ----------------------------------------------------------------------
 size of atom nlocal's restart data
 ------------------------------------------------------------------------- */

void CreateSmdTriBoundary::read_triangles(int pass) {

	double coord[3];

	int nlocal_previous = atom->nlocal;
	int ilocal = nlocal_previous;
	int m;
	int me;

	Vector3d *vert;
	vert = new Vector3d[3];
	Vector3d normal, center;

	FILE *fp = fopen(filename.c_str(), "r");
	if (fp == NULL) {
		char str[128];
		sprintf(str, "Cannot open file %s", filename.c_str());
		error->one(FLERR, str);
	}

	MPI_Comm_rank(world, &me);
	if (me == 0) {
		if (screen) {
				fprintf(screen, "... reading triangles ...\n");
		}
		if (logfile) {
				fprintf(logfile, "... reading triangles ...\n");
		}
	}

	char str[128];
	char line[256];
	char *retpointer;
	char **values;
	int nwords;

	// read STL solid name
	retpointer = fgets(line, sizeof(line), fp);
	if (retpointer == NULL) {
		sprintf(str, "error reading number of triangle pairs");
		error->one(FLERR, str);
	}

	nwords = count_words(line);
	if (nwords < 1) {
		sprintf(str, "first line of file is incorrect");
		error->one(FLERR, str);
	}

	// iterate over STL facets util end of body is reached

	while (fgets(line, sizeof(line), fp)) { // read a line, should be the facet line

		// evaluate facet line
		nwords = count_words(line);
		if (nwords != 5) {
			//sprintf(str, "found end solid line");
			//error->message(FLERR, str);
			break;
		} else {
			// should be facet line
		}

		values = new char*[nwords];
		values[0] = strtok(line, " \t\n\r\f");
		if (values[0] == NULL)
			error->all(FLERR, "Incorrect atom format in data file");
		for (m = 1; m < nwords; m++) {
			values[m] = strtok(NULL, " \t\n\r\f");
			if (values[m] == NULL)
				error->all(FLERR, "Incorrect atom format in data file");
		}

		normal << force->numeric(FLERR, values[2]), force->numeric(FLERR, values[3]), force->numeric(FLERR, values[4]);
		//cout << "normal is " << normal << endl;

		delete[] values;

		// read outer loop line
		retpointer = fgets(line, sizeof(line), fp);
		if (retpointer == NULL) {
			sprintf(str, "error reading outer loop");
			error->one(FLERR, str);
		}

		nwords = count_words(line);
		if (nwords != 2) {
			sprintf(str, "error reading outer loop");
			error->one(FLERR, str);
		}

		// read vertex lines

		for (int k = 0; k < 3; k++) {
			retpointer = fgets(line, sizeof(line), fp);
			if (retpointer == NULL) {
				sprintf(str, "error reading vertex line");
				error->one(FLERR, str);
			}

			nwords = count_words(line);
			if (nwords != 4) {
				sprintf(str, "error reading vertex line");
				error->one(FLERR, str);
			}

			values = new char*[nwords];
			values[0] = strtok(line, " \t\n\r\f");
			if (values[0] == NULL)
				error->all(FLERR, "Incorrect vertex line");
			for (m = 1; m < nwords; m++) {
				values[m] = strtok(NULL, " \t\n\r\f");
				if (values[m] == NULL)
					error->all(FLERR, "Incorrect vertex line");
			}

			vert[k] << force->numeric(FLERR, values[1]), force->numeric(FLERR, values[2]), force->numeric(FLERR, values[3]);
			//cout << "vertex is " << vert[k] << endl;
			//printf("%s %s %s\n", values[1], values[2], values[3]);
			delete[] values;
			//exit(1);

		}

		// read end loop line
		retpointer = fgets(line, sizeof(line), fp);
		if (retpointer == NULL) {
			sprintf(str, "error reading endloop");
			error->one(FLERR, str);
		}

		nwords = count_words(line);
		if (nwords != 1) {
			sprintf(str, "error reading endloop");
			error->one(FLERR, str);
		}

		// read end facet line
		retpointer = fgets(line, sizeof(line), fp);
		if (retpointer == NULL) {
			sprintf(str, "error reading endfacet");
			error->one(FLERR, str);
		}

		nwords = count_words(line);
		if (nwords != 1) {
			sprintf(str, "error reading endfacet");
			error->one(FLERR, str);
		}

		// now we have a normal and three vertices ... proceed with adding triangle

		center = (vert[0] + vert[1] + vert[2]) / 3.0;

		//	cout << "center is " << center << endl;

		double r1 = (center - vert[0]).norm();
		double r2 = (center - vert[1]).norm();
		double r3 = (center - vert[2]).norm();
		double r = MAX(r1, r2);
		r = MAX(r, r3);

		/*
		 * if atom/molecule is in my subbox, create it
		 * ... use x0 to hold triangle normal.
		 * ... use smd_data_9 to hold the three vertices
		 * ... use x to hold triangle center
		 * ... radius is the mmaximal distance from triangle center to all vertices
		 */

		//	printf("coord: %f %f %f\n", coord[0], coord[1], coord[2]);
		//	printf("sublo: %f %f %f\n", sublo[0], sublo[1], sublo[2]);
		//	printf("subhi: %f %f %f\n", subhi[0], subhi[1], subhi[2]);
		//printf("ilocal = %d\n", ilocal);
		if (center(0) >= sublo[0] && center(0) < subhi[0] && center(1) >= sublo[1] && center(1) < subhi[1] && center(2) >= sublo[2]
				&& center(2) < subhi[2]) {
			//printf("******* KERATIN nlocal=%d ***\n", nlocal);
			coord[0] = center(0);
			coord[1] = center(1);
			coord[2] = center(2);
			atom->avec->create_atom(wall_particle_type, coord);

			/*
			 * need to initialize pointers to atom vec arrays here, because they could have changed
			 * due to calling grow() in create_atoms() above;
			 */

			int *mol = atom->molecule;
			int *type = atom->type;
			double *radius = atom->radius;
			double *contact_radius = atom->contact_radius;
			double **smd_data_9 = atom->smd_data_9;
			double **x0 = atom->x0;

			radius[ilocal] = r;
			contact_radius[ilocal] = r;
			mol[ilocal] = wall_molecule_id;
			type[ilocal] = wall_particle_type;
			x0[ilocal][0] = normal(0);
			x0[ilocal][1] = normal(1);
			x0[ilocal][2] = normal(2);
			smd_data_9[ilocal][0] = vert[0](0);
			smd_data_9[ilocal][1] = vert[0](1);
			smd_data_9[ilocal][2] = vert[0](2);
			smd_data_9[ilocal][3] = vert[1](0);
			smd_data_9[ilocal][4] = vert[1](1);
			smd_data_9[ilocal][5] = vert[1](2);
			smd_data_9[ilocal][6] = vert[2](0);
			smd_data_9[ilocal][7] = vert[2](1);
			smd_data_9[ilocal][8] = vert[2](2);

			ilocal++;
		}

	}

	delete[] vert;
	fclose(fp);
}
