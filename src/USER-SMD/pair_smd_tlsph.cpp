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

#include "group.h"
#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "pair_smd_tlsph.h"
#include "fix_smd_tlsph_reference_configuration.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <stdio.h>
#include <iostream>
#include "math_special.h"
#include <map>
#include "update.h"
#include <Eigen/SVD>
#include <Eigen/Eigen>
#include "smd_material_models.h"
#include "smd_kernels.h"
#include "smd_math.h"
using namespace SMD_Kernels;
using namespace Eigen;
using namespace std;
using namespace LAMMPS_NS;
using namespace SMD_Math;

#define JAUMANN false
#define DETF_MIN 0.01 // maximum compression deformation allowed
#define DETF_MAX 40.0 // maximum tension deformation allowed
#define TLSPH_DEBUG 0
#define PLASTIC_STRAIN_AVERAGE_WINDOW 100.0

/* ---------------------------------------------------------------------- */

PairTlsph::PairTlsph(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;

	strengthModel = eos = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	Fdot = Fincr = K = PK1 = NULL;
	d = R = FincrInv = W = D = NULL;
	detF = NULL;
	smoothVelDifference = NULL;
	shearFailureFlag = NULL;
	numNeighsRefConfig = NULL;
	shepardWeight = NULL;
	CauchyStress = NULL;
	hourglass_error = NULL;
	Lookup = NULL;

	updateFlag = 0;
	not_first = 0;

	comm_forward = 21; // this pair style communicates 20 doubles to ghost atoms : PK1 tensor + F tensor + shepardWeight
	fix_tlsph_reference_configuration = NULL;

	cut_comm = MAX(neighbor->cutneighmax, comm->cutghostuser); // cutoff radius within which ghost atoms are communicated.
}

/* ---------------------------------------------------------------------- */

PairTlsph::~PairTlsph() {
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(strengthModel);
		memory->destroy(eos);
		memory->destroy(Lookup);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;

		delete[] Fdot;
		delete[] Fincr;
		delete[] K;
		delete[] detF;
		delete[] PK1;
		delete[] smoothVelDifference;
		delete[] d;
		delete[] R;
		delete[] FincrInv;
		delete[] W;
		delete[] D;
		delete[] shearFailureFlag;
		delete[] numNeighsRefConfig;
		delete[] shepardWeight;
		delete[] CauchyStress;
		delete[] hourglass_error;
	}
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairTlsph::PreCompute() {
	int *mol = atom->molecule;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double **x0 = atom->x0;
	double **x = atom->x;
	double **v = atom->vest; // extrapolated velocities corresponding to current positions
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	int *tag = atom->tag;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int jnum, jj, i, j, itype, idim;

	tagint **partner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partner;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
	float **wfd_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wfd_list;
	float **wf_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wf_list;
	double r0, r0Sq, wf, wfd, h, irad, voli, volj;
	Vector3d dx0, dx, dv, g;
	Matrix3d Ktmp, Fdottmp, Ftmp, L, Fold, U, eye;
	Vector3d xi, xj, vi, vj, vinti, vintj, x0i, x0j, dvint, du;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	eye.setIdentity();

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		K[i].setZero();
		Fincr[i].setZero();
		Fdot[i].setZero();
		shearFailureFlag[i] = false;
		shepardWeight[i] = 0.0;
		numNeighsRefConfig[i] = 0;
		smoothVelDifference[i].setZero();
		hourglass_error[i] = 0.0;
	}

	for (i = 0; i < nlocal; i++) {

		if (mol[i] < 0) { // valid SPH particle have mol > 0
			continue;
		}

		jnum = npartner[i];
		irad = radius[i];
		voli = vfrac[i];

		// initialize Eigen data structures from LAMMPS data structures
		for (idim = 0; idim < 3; idim++) {
			x0i(idim) = x0[i][idim];
			xi(idim) = x[i][idim];
			vi(idim) = v[i][idim];
			vinti(idim) = vint[i][idim];
		}

//		x0i << x0[i][0], x0[i][1], x0[i][2];
//		xi << x[i][0], x[i][1], x[i][2];
//		vi << v[i][0], v[i][1], v[i][2];
//		vinti << vint[i][0], vint[i][1], vint[i][2];

		for (jj = 0; jj < jnum; jj++) {

			if (partner[i][jj] == 0)
				continue;
			j = atom->map(partner[i][jj]);
			if (j < 0) { //			// check if lost a partner without first breaking bond
				partner[i][jj] = 0;
				continue;
			}

			if (mol[j] < 0) { // particle has failed. do not include it for computing any property
				continue;
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			// initialize Eigen data structures from LAMMPS data structures
			for (idim = 0; idim < 3; idim++) {
				x0j(idim) = x0[j][idim];
				xj(idim) = x[j][idim];
				vj(idim) = v[j][idim];
				vintj(idim) = vint[j][idim];
			}
			//			xj << x[j][0], x[j][1], x[j][2];
			//			vj << v[j][0], v[j][1], v[j][2];
			//			vintj << vint[j][0], vint[j][1], vint[j][2];
			//x0j << x0[j][0], x0[j][1], x0[j][2];
			dx0 = x0j - x0i;

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			r0Sq = dx0.squaredNorm();
			h = irad + radius[j];

			r0 = sqrt(r0Sq);
			volj = vfrac[j];

			// distance vectors in current and reference configuration, velocity difference
			dx = xj - xi;
			if (periodic)
				domain->minimum_image(dx(0), dx(1), dx(2));
			dv = vj - vi;
			dvint = vintj - vinti;

			//barbara_kernel_and_derivative(h, r0, domain->dimension, wf, wfd);
			//spiky_kernel_and_derivative(h, r0, wf, wfd);
			wf = wf_list[i][jj];
			wfd = wfd_list[i][jj];
			g = (wfd / r0) * dx0;

			/* build matrices */
			Ktmp = g * dx0.transpose();
			Ftmp = (dx - dx0) * g.transpose();
			Fdottmp = dv * g.transpose();

			K[i] += volj * Ktmp;
			Fincr[i] += volj * Ftmp;
			Fdot[i] += volj * Fdottmp;
			shepardWeight[i] += volj * wf;
			smoothVelDifference[i] += volj * wf * dvint;
			numNeighsRefConfig[i]++;
		} // end loop over j
	} // end loop over i

	/*
	 * invert shape matrix and compute corrected quantities
	 */

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype] == 1) { // we only do the subsequent calculation if pair style is defined for this particle

			if ((numNeighsRefConfig[i] < domain->dimension + 2) && (mol[i] > 0)) { // cannot possibly invert shape matrix
				printf("deleting particle [%d] because number of neighbors=%d is too small\n", tag[i], numNeighsRefConfig[i]);
				mol[i] = -1;
			}

			if (mol[i] > 0) {

				if (domain->dimension == 2) {
					K[i](0, 2) = 0.0;
					K[i](1, 2) = 0.0;
					K[i](2, 0) = 0.0;
					K[i](2, 1) = 0.0;
					K[i](2, 2) = 1.0;
					K[i] = K[i].inverse().eval();
					//K[i] = PairTlsph::pseudo_inverse_SVD(K[i]);
					K[i](2, 2) = 1.0; // make inverse of K well defined even when it is rank-deficient (3d matrix, only 2d information)
				} else {
					K[i] = K[i].inverse().eval();
					//K[i] = PairTlsph::pseudo_inverse_SVD(K[i]);
				}

				//K[i].setIdentity();

				Fincr[i] *= K[i]; // use shape matrix to obtain first-order corrected SPH approximation
				Fincr[i] += eye;
				Fdot[i] *= K[i];

				/*
				 * we need to be able to recover from a potentially planar (2d) configuration of particles
				 */
				if (domain->dimension == 2) {
					Fincr[i](0, 2) = 0.0;
					Fincr[i](1, 2) = 0.0;
					Fincr[i](2, 0) = 0.0;
					Fincr[i](2, 1) = 0.0;
					Fincr[i](2, 2) = 1.0;
					Fdot[i](0, 2) = 0.0;
					Fdot[i](1, 2) = 0.0;
					Fdot[i](2, 0) = 0.0;
					Fdot[i](2, 1) = 0.0;
					Fdot[i](2, 2) = 0.0;
				}
				detF[i] = Fincr[i].determinant();
				//FincrInv[i] = PairTlsph::pseudo_inverse_SVD(Fincr[i]);;
				FincrInv[i] = Fincr[i].inverse();

				/*
				 * check that F * F^-1 is unit matrix
				 */
				Matrix3d Mdiff = Fincr[i] * FincrInv[i] - eye;
				if (Mdiff.norm() > 1.0e-10) {
					printf("Inversion of F is inaccurate\n");
					cout << "Here is the difference matrix to unity:" << endl << Mdiff << endl;
				}

//				if (tag[i] == 101) {
//					printf("J=%f, nn = %d\n", detF[i], numNeighsRefConfig[i]);
//					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
//					cout << "Here is matrix F-1:" << endl << FincrInv[i] << endl;
//					cout << "Here is matrix K-1:" << endl << K[i] << endl << endl;
//				}

				/*
				 * make sure F stays within some limits
				 */

				if (detF[i] < DETF_MIN) {
					printf("deleting particle [%d] because det(F)=%f is smaller than limit=%f\n", tag[i], Fincr[i].determinant(),
					DETF_MIN);
					printf("nn = %d\n", numNeighsRefConfig[i]);
					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
					cout << "Here is matrix F-1:" << endl << FincrInv[i] << endl;
					cout << "Here is matrix K-1:" << endl << K[i] << endl;
					mol[i] = -1;
					//error->one(FLERR, "");
				} else if (detF[i] > DETF_MAX) {
					printf("deleting particle [%d] because det(F)=%f is larger than limit=%f\n", tag[i], Fincr[i].determinant(),
					DETF_MAX);
					printf("nn = %d\n", numNeighsRefConfig[i]);
					cout << "Here is matrix F:" << endl << Fincr[i] << endl;
					cout << "Here is matrix F-1:" << endl << FincrInv[i] << endl;
					cout << "Here is matrix K-1:" << endl << K[i] << endl;
					mol[i] = -1;
					//error->one(FLERR, "");
				}

				if (mol[i] > 0) {
					// velocity gradient, see Pronto2d, eqn.(2.1.3)
					L = Fdot[i] * FincrInv[i];

					// symmetric (D) and asymmetric (W) parts of L
					D[i] = 0.5 * (L + L.transpose());
					W[i] = 0.5 * (L - L.transpose());					// spin tensor:: need this for Jaumann rate

					if (JAUMANN) {
						d[i] = D[i];
						R[i].setIdentity(); // for Jaumann stress rate, we do not need a subsequent rotation back into the reference configuration
					} else {

						// convention: unrotated frame is that one, where the true rotation of an integration point has been subtracted.
						// stress in the unrotated frame of reference is denoted sigma (stress seen by an observer doing rigid body rotations along with the material)
						// stress in the true frame of reference (a stationary observer) is denoted by T, "true stress"

						// polar decomposition of the deformation gradient, F = R * U

						PairTlsph::PolDec(Fincr[i], &R[i], &U);

						// unrotated rate-of-deformation tensor d, see right side of Pronto2d, eqn.(2.1.7)
						d[i] = R[i].transpose() * D[i] * R[i];
					}

					// normalize average velocity field around an integration point
					if (shepardWeight[i] > 0.0) {
						smoothVelDifference[i] /= shepardWeight[i];
					} else {
						smoothVelDifference[i].setZero();
					}

				} // end if mol[i] > 0

			} // end if mol[i] > 0

			if (mol[i] < 0) {

				d[i].setZero();
				D[i].setZero();
				Fdot[i].setZero();
				Fincr[i].setIdentity();
				//smoothVel[i].setZero();
				detF[i] = 1.0;
				K[i].setIdentity();
			}
		}  // end if setflag[itype]
	} // end loop over i = 0 to nlocal
}

/* ---------------------------------------------------------------------- */

void PairTlsph::compute(int eflag, int vflag) {

	if (atom->nmax > nmax) {
		nmax = atom->nmax;
		delete[] Fdot;
		Fdot = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] Fincr;
		Fincr = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] K;
		K = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] PK1;
		PK1 = new Matrix3d[nmax]; // memory usage: 9 doubles; total 5*9=45 doubles
		delete[] detF;
		detF = new double[nmax]; // memory usage: 1 double; total 46 doubles
		delete[] smoothVelDifference;
		smoothVelDifference = new Vector3d[nmax]; // memory usage: 3 doubles; total 49 doubles
		delete[] d;
		d = new Matrix3d[nmax]; // memory usage: 9 doubles; total 58 doubles
		delete[] R;
		R = new Matrix3d[nmax]; // memory usage: 9 doubles; total 67 doubles
		delete[] FincrInv;
		FincrInv = new Matrix3d[nmax]; // memory usage: 9 doubles; total 85 doubles
		delete[] W;
		W = new Matrix3d[nmax]; // memory usage: 9 doubles; total 94 doubles
		delete[] D;
		D = new Matrix3d[nmax]; // memory usage: 9 doubles; total 103 doubles
		delete[] shearFailureFlag;
		shearFailureFlag = new bool[nmax]; // memory usage: 1 double; total 104 doubles
		delete[] numNeighsRefConfig;
		numNeighsRefConfig = new int[nmax]; // memory usage: 1 int; total 108 doubles
		delete[] shepardWeight;
		shepardWeight = new double[nmax]; // memory usage: 1 double; total 109 doubles
		delete[] CauchyStress;
		CauchyStress = new Matrix3d[nmax]; // memory usage: 9 doubles; total 118 doubles
		delete[] hourglass_error;
		hourglass_error = new double[nmax];
	}

	if (update->ntimestep == 0) { // return on zeroth timestep because reference connectivity lists still needs to be built
		return;
	}

	PairTlsph::PreCompute();
	PairTlsph::AssembleStress();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);

	ComputeForces(eflag, vflag);
}

void PairTlsph::ComputeForces(int eflag, int vflag) {
	int *mol = atom->molecule;
	double **x = atom->x;
	double **v = atom->vest;
	double **x0 = atom->x0;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *de = atom->de;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *plastic_strain = atom->eff_plastic_strain;
	double *damage = atom->damage;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, jj, jnum, itype, idim;
	double r, hg_mag, wf, wfd, h, r0, r0Sq, voli, volj;
	double delVdotDelR, visc_magnitude, deltaE, mu_ij, hg_err, gamma_dot_dx, delta;
	char str[128];
	Vector3d fi, fj, dx0, dx, dv, f_stress, f_hg, dxp_i, dxp_j, gamma, g, gamma_i, gamma_j, x0i, x0j;
	Vector3d xi, xj, vi, vj, f_visc, sumForces;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	tagint **partner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partner;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
	float **wfd_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wfd_list;
	float **wf_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wf_list;

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	/*
	 * iterate over pairs of particles i, j and assign forces using PK1 stress tensor
	 */

	updateFlag = 0;
	hMin = 1.0e22;
	dtRelative = 1.0e22;

	for (i = 0; i < nlocal; i++) {
		if (mol[i] < 0) {
			continue; // Particle i is not a valid SPH particle (anymore). Skip all interactions with this particle.
		}

		itype = type[i];
		jnum = npartner[i];
		voli = vfrac[i];

		for (idim = 0; idim < 3; idim++) {
			x0i(idim) = x0[i][idim];
			xi(idim) = x[i][idim];
			vi(idim) = v[i][idim];
		}

		for (jj = 0; jj < jnum; jj++) {
			if (partner[i][jj] == 0)
				continue;
			j = atom->map(partner[i][jj]);
			if (j < 0) { //			// check if lost a partner without first breaking bond
				partner[i][jj] = 0;
				continue;
			}

			if (mol[j] < 0) {
				continue; // Particle j is not a valid SPH particle (anymore). Skip all interactions with this particle.
			}

			if (mol[i] != mol[j]) {
				continue;
			}

			if (type[j] != itype) {
				sprintf(str, "particle pair is not of same type!");
				error->all(FLERR, str);
			}

			for (idim = 0; idim < 3; idim++) {
				x0j(idim) = x0[j][idim];
				xj(idim) = x[j][idim];
				vj(idim) = v[j][idim];
			}

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			// check that distance between i and j (in the reference config) is less than cutoff
			dx0 = x0j - x0i;

			r0Sq = dx0.squaredNorm();
			h = radius[i] + radius[j];
			hMin = MIN(hMin, h);
			r0 = sqrt(r0Sq);
			volj = vfrac[j];

			// distance vectors in current and reference configuration, velocity difference
			dx = xj - xi;
			dv = vj - vi;

			r = dx.norm(); // current distance

			//barbara_kernel_and_derivative(h, r0, domain->dimension, wf, wfd);
			//spiky_kernel_and_derivative(h, r0, wf, wfd);
			wf = wf_list[i][jj];
			wfd = wfd_list[i][jj];
			g = (wfd / r0) * dx0; // uncorrected kernel gradient

			/*
			 * force contribution -- note that the kernel gradient correction has been absorbed into PK1
			 */
			f_stress = voli * volj * (PK1[i] + PK1[j]) * g;

			/*
			 * artificial viscosity
			 */
			delVdotDelR = dx.dot(dv) / (r + 0.1 * h); // project relative velocity onto unit particle distance vector [m/s]
			LimitDoubleMagnitude(delVdotDelR, 0.01 * Lookup[SIGNAL_VELOCITY][itype]);
			mu_ij = h * delVdotDelR / (r + 0.1 * h); // units: [m * m/s / m = m/s]
			visc_magnitude = (-Lookup[VISCOSITY_Q1][itype] * Lookup[SIGNAL_VELOCITY][itype] * mu_ij
					+ Lookup[VISCOSITY_Q2][itype] * mu_ij * mu_ij) / Lookup[REFERENCE_DENSITY][itype]; // units: m^5/(s^2 kg))
			f_visc = rmass[i] * rmass[j] * visc_magnitude * wfd * dx / (r + 1.0e-2 * h); // units: kg^2 * m^5/(s^2 kg) * m^-4 = kg m / s^2 = N

			/*
			 * hourglass deviation of particles i and j
			 */

			gamma = 0.5 * (Fincr[i] + Fincr[j]) * dx0 - dx;
			hg_err = gamma.norm() / r0;
			hourglass_error[i] += volj * wf * hg_err;

			/* SPH-like hourglass formulation */

			if (MAX(plastic_strain[i], plastic_strain[j]) > 1.0e-3) {
				/*
				 * viscous hourglass formulation for particles with plastic deformation
				 */
				delta = gamma.dot(dx);
				if (delVdotDelR * delta < 0.0) {
					hg_err = MAX(hg_err, 0.1); // limit hg_err to avoid numerical instabilities
					hg_mag = -hg_err * Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * Lookup[SIGNAL_VELOCITY][itype] * mu_ij
							/ Lookup[REFERENCE_DENSITY][itype]; // this has units of pressure
				} else {
					hg_mag = 0.0;
				}
				f_hg = rmass[i] * rmass[j] * hg_mag * wfd * dx / (r + 1.0e-2 * h);

			} else {
				/*
				 * stiffness hourglass formulation for particle in the elastic regime
				 */

				gamma_dot_dx = gamma.dot(dx); // project hourglass error vector onto pair distance vector
				LimitDoubleMagnitude(gamma_dot_dx, 0.1 * r); // limit projected vector to avoid numerical instabilities
				delta = 0.5 * gamma_dot_dx / (r + 0.1 * h); // delta has dimensions of [m]
				hg_mag = Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * delta / (r0Sq + 0.01 * h * h); // hg_mag has dimensions [m^(-1)]
				hg_mag *= -voli * volj * wf * Lookup[YOUNGS_MODULUS][itype]; // hg_mag has dimensions [J*m^(-1)] = [N]
				f_hg = (hg_mag / (r + 0.01 * h)) * dx;
			}

			f_hg *= (1.0 - MAX(damage[i], damage[j])); // reduce hourglass force between damaged particles

			// sum stress, viscous, and hourglass forces
			sumForces = f_stress + f_visc + f_hg;

			// energy rate -- project velocity onto force vector
			deltaE = 0.5 * sumForces.dot(dv);

			// apply forces to pair of particles
			f[i][0] += sumForces(0);
			f[i][1] += sumForces(1);
			f[i][2] += sumForces(2);
			de[i] += deltaE;

			// tally atomistic stress tensor
			if (evflag) {
				ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
			}

			// check if a particle has moved too much w.r.t another particle
			if (r > r0) {
				if (update_method == UPDATE_CONSTANT_THRESHOLD) {
					if (r - r0 > update_threshold) {
						updateFlag = 1;
					}
				} else if (update_method == UPDATE_PAIRWISE_RATIO) {
					if ((r - r0) / h > update_threshold) {
						updateFlag = 1;
					}
				}
			}
		}
	}

	for (i = 0; i < nlocal; i++) {
		if (shepardWeight[i] != 0.0) {
			hourglass_error[i] /= shepardWeight[i];
		}
	}

	if (vflag_fdotr)
		virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 assemble unrotated stress tensor using deviatoric and pressure components.
 Convert to corotational Cauchy stress, then to PK1 stress and apply
 shape matrix correction
 ------------------------------------------------------------------------- */
void PairTlsph::AssembleStress() {
	int *mol = atom->molecule;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	double **tlsph_stress = atom->smd_stress;
	int *type = atom->type;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double *rmass = atom->rmass;
	double *vfrac = atom->vfrac;
	double *e = atom->e;
	double pInitial, d_iso, pFinal, p_rate, plastic_strain_increment;
	int i, itype;
	int nlocal = atom->nlocal;
	double dt = update->dt;
	double M_eff, p_wave_speed, mass_specific_energy, vol_specific_energy, rho, damage_increment;
	Matrix3d sigma_rate, eye, sigmaInitial, sigmaFinal, T, Jaumann_rate, sigma_rate_check;
	Matrix3d d_dev, sigmaInitial_dev, sigmaFinal_dev, sigma_dev_rate, strain, sigma_damaged;
	Vector3d x0i, xi, xp;

	eye.setIdentity();
	dtCFL = 1.0e22;
	pFinal = 0.0;

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype] == 1) {
			if (mol[i] > 0) { // only do the following if particle has not failed -- mol < 0 means particle has failed

				/*
				 * initial stress state: given by the unrotateted Cauchy stress.
				 * Assemble Eigen 3d matrix from stored stress state
				 */
				sigmaInitial(0, 0) = tlsph_stress[i][0];
				sigmaInitial(0, 1) = tlsph_stress[i][1];
				sigmaInitial(0, 2) = tlsph_stress[i][2];
				sigmaInitial(1, 1) = tlsph_stress[i][3];
				sigmaInitial(1, 2) = tlsph_stress[i][4];
				sigmaInitial(2, 2) = tlsph_stress[i][5];
				sigmaInitial(1, 0) = sigmaInitial(0, 1);
				sigmaInitial(2, 0) = sigmaInitial(0, 2);
				sigmaInitial(2, 1) = sigmaInitial(1, 2);

				pInitial = sigmaInitial.trace() / 3.0; // initial pressure, isotropic part of initial stress
				sigmaInitial_dev = PairTlsph::Deviator(sigmaInitial);
				d_iso = d[i].trace(); // volumetric part of stretch rate
				d_dev = PairTlsph::Deviator(d[i]); // deviatoric part of stretch rate
				strain = 0.5 * (Fincr[i].transpose() * Fincr[i] - eye);

				mass_specific_energy = e[i] / rmass[i]; // energy per unit mass
				rho = rmass[i] / (detF[i] * vfrac[i]); // current mass density
				vol_specific_energy = mass_specific_energy * rho; // energy per current volume

				/*
				 * pressure: compute pressure rate p_rate and final pressure pFinal
				 */

				ComputePressure(i, pInitial, d_iso, pFinal, p_rate);

				/*
				 * material strength
				 */

				if (damage[i] < 1.0) {
					ComputeStressDeviator(i, sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);

					eff_plastic_strain[i] += plastic_strain_increment;

					eff_plastic_strain_rate[i] -= eff_plastic_strain_rate[i] / PLASTIC_STRAIN_AVERAGE_WINDOW;
					eff_plastic_strain_rate[i] += (plastic_strain_increment / dt) / PLASTIC_STRAIN_AVERAGE_WINDOW;
				} else {
					sigmaFinal_dev.setZero(); // damage[i] >= 1, therefore it cannot carry shear stress
					sigma_dev_rate.setZero();
				}

				/*
				 *  assemble total stress from pressure and deviatoric stress
				 */
				sigmaFinal = pFinal * eye + sigmaFinal_dev; // this is the stress that is kept

				/*
				 *  Damage due to failure criteria.
				 */

				if (damage[i] < 1.0) {

					ComputeDamage(i, strain, pFinal, sigmaFinal, sigmaFinal_dev, sigma_damaged, damage_increment);

					damage[i] += damage_increment;
					damage[i] = MIN(damage[i], 1.0);

				} // end if damage[i] < 1.0

				/*
				 * Apply damage
				 */

				if (pFinal < 0.0) { // compression: particle can carry compressive load but reduced shear
					sigma_damaged = pFinal * eye + (1.0 - damage[i]) * sigmaFinal_dev;
				} else { // tension: particle has reduced tensile and shear load bearing capability
					sigma_damaged = (1.0 - damage[i]) * (pFinal * eye + sigmaFinal_dev);
				}

				if (JAUMANN) {
					/*
					 * sigma is already the co-rotated Cauchy stress.
					 * The stress rate, however, needs to be made objective.
					 */

					if (dt > 1.0e-16) {
						sigma_rate = (1.0 / dt) * (sigmaFinal - sigmaInitial);
					} else {
						sigma_rate.setZero();
					}

					Jaumann_rate = sigma_rate + W[i] * sigmaInitial + sigmaInitial * W[i].transpose();
					sigmaFinal = sigmaInitial + dt * Jaumann_rate;
					T = sigmaFinal;
				} else {
					/*
					 * sigma is the unrotated stress.
					 * need to do forward rotation of the unrotated stress sigma to the current configuration
					 */
					T = R[i] * sigma_damaged * R[i].transpose();
				}

				/*
				 * store unrotated stress in atom vector
				 * symmetry is exploited
				 */
				tlsph_stress[i][0] = sigmaFinal(0, 0);
				tlsph_stress[i][1] = sigmaFinal(0, 1);
				tlsph_stress[i][2] = sigmaFinal(0, 2);
				tlsph_stress[i][3] = sigmaFinal(1, 1);
				tlsph_stress[i][4] = sigmaFinal(1, 2);
				tlsph_stress[i][5] = sigmaFinal(2, 2);

				// store rotated, "true" Cauchy stress
				CauchyStress[i] = T;

				/*
				 * We have the corotational Cauchy stress.
				 * Convert to PK1. Note that reference configuration used for computing the forces is linked via
				 * the incremental deformation gradient, not the full deformation gradient.
				 */
				PK1[i] = detF[i] * T * FincrInv[i].transpose();

				/*
				 * pre-multiply stress tensor with shape matrix to save computation in force loop
				 */
				PK1[i] = PK1[i] * K[i];

				//cout << "F:" << Fincr[i] << endl;
				//cout << "T:" << T << endl;
				//cout << "PK1:" << PK1[i] << endl << endl;

				/*
				 * compute stable time step according to Pronto 2d
				 */

				Matrix3d deltaSigma;
				deltaSigma = sigma_damaged - sigmaInitial;
				p_rate = deltaSigma.trace() / (3.0 * dt);
				sigma_dev_rate = Deviator(deltaSigma) / dt;

				double K_eff, mu_eff;
				effective_longitudinal_modulus(itype, dt, d_iso, p_rate, d_dev, sigma_dev_rate, damage[i], K_eff, mu_eff, M_eff);
				p_wave_speed = sqrt(M_eff / rho);
				dtCFL = MIN(dtCFL, 2.0 * radius[i] / p_wave_speed);

				//dtCFL = MIN(dtCFL, radius[i] / p_wave_speed);

			} else { // end if delete_flag == 0
				PK1[i].setZero();
				K[i].setIdentity();
				CauchyStress[i].setZero();

				sigma_rate.setZero();
				tlsph_stress[i][0] = 0.0;
				tlsph_stress[i][1] = 0.0;
				tlsph_stress[i][2] = 0.0;
				tlsph_stress[i][3] = 0.0;
				tlsph_stress[i][4] = 0.0;
				tlsph_stress[i][5] = 0.0;
			}

		} else { // setflag[i] is not checked
			PK1[i].setZero();
			K[i].setIdentity();
			CauchyStress[i].setZero();
		}
	}
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairTlsph::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(strengthModel, n + 1, "pair:strengthmodel");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(Lookup, MAX_KEY_VALUE, n + 1, "pair:LookupTable");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairTlsph::settings(int narg, char **arg) {

	if (comm->me == 0) {
		printf(
				"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("TLSPH settings\n");
	}

	/*
	 * default value for update_threshold for updates of reference configuration:
	 * The maximum relative displacement which is tracked by the construction of LAMMPS' neighborlists
	 * is the folowing.
	 */

	cut_comm = MAX(neighbor->cutneighmax, comm->cutghostuser); // cutoff radius within which ghost atoms are communicated.
	update_threshold = cut_comm;
	update_method = UPDATE_NONE;

	int iarg = 0;

	while (true) {

		if (iarg >= narg) {
			break;
		}

		if (strcmp(arg[iarg], "*UPDATE_CONSTANT") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following *UPDATE_CONSTANT keyword");
			}

			update_method = UPDATE_CONSTANT_THRESHOLD;
			update_threshold = force->numeric(FLERR, arg[iarg]);

		} else if (strcmp(arg[iarg], "*UPDATE_PAIRWISE") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following *UPDATE_PAIRWISE keyword");
			}

			update_method = UPDATE_PAIRWISE_RATIO;
			update_threshold = force->numeric(FLERR, arg[iarg]);

		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for smd/integrate_tlsph: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;
	}

	if ((update_threshold > cut_comm) && (update_method == UPDATE_CONSTANT_THRESHOLD)) {
		if (comm->me == 0) {
			printf("\n                ***** WARNING ***\n");
			printf("requested reference configuration update threshold is %g length units\n", update_threshold);
			printf("This value exceeds the maximum value %g beyond which TLSPH displacements can be tracked at current settings.\n",
					cut_comm);
			printf("Expect loss of neighbors!\n");
		}
	}

	if (comm->me == 0) {

		if (update_method == UPDATE_CONSTANT_THRESHOLD) {
			printf("... will update reference configuration if magnitude of relative displacement exceeds %g length units\n",
					update_threshold);
		} else if (update_method == UPDATE_PAIRWISE_RATIO) {
			printf("... will update reference configuration if ratio pairwise distance / smoothing length  exceeds %g\n",
					update_threshold);
		} else if (update_method == UPDATE_NONE) {
			printf("... will never update reference configuration");
		}
		printf(
				">>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");

	}

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairTlsph::coeff(int narg, char **arg) {
	int ioffset, iarg, iNextKwd, itype;
	char str[128];
	std::string s, t;

	if (narg < 3) {
		sprintf(str, "number of arguments for pair tlsph is too small!");
		error->all(FLERR, str);
	}
	if (!allocated)
		allocate();

	/*
	 * check that TLSPH parameters are given only in i,i form
	 */
	if (force->inumeric(FLERR, arg[0]) != force->inumeric(FLERR, arg[1])) {
		sprintf(str, "TLSPH coefficients can only be specified between particles of same type!");
		error->all(FLERR, str);
	}
	itype = force->inumeric(FLERR, arg[0]);

	if (comm->me == 0) {
		printf(
				"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("SMD / TLSPH PROPERTIES OF PARTICLE TYPE %d:\n", itype);
	}

	/*
	 * read parameters which are common -- regardless of material / eos model
	 */

	ioffset = 2;
	if (strcmp(arg[ioffset], "*COMMON") != 0) {
		sprintf(str, "common keyword missing!");
		error->all(FLERR, str);
	}

	t = string("*");
	iNextKwd = -1;
	for (iarg = ioffset + 1; iarg < narg; iarg++) {
		s = string(arg[iarg]);
		if (s.compare(0, t.length(), t) == 0) {
			iNextKwd = iarg;
			break;
		}
	}

	//printf("keyword following *COMMON is %s\n", arg[iNextKwd]);

	if (iNextKwd < 0) {
		sprintf(str, "no *KEYWORD terminates *COMMON");
		error->all(FLERR, str);
	}

	if (iNextKwd - ioffset != 7 + 1) {
		sprintf(str, "expected 7 arguments following *COMMON but got %d\n", iNextKwd - ioffset - 1);
		error->all(FLERR, str);
	}

	Lookup[REFERENCE_DENSITY][itype] = force->numeric(FLERR, arg[ioffset + 1]);
	Lookup[YOUNGS_MODULUS][itype] = force->numeric(FLERR, arg[ioffset + 2]);
	Lookup[POISSON_RATIO][itype] = force->numeric(FLERR, arg[ioffset + 3]);
	Lookup[VISCOSITY_Q1][itype] = force->numeric(FLERR, arg[ioffset + 4]);
	Lookup[VISCOSITY_Q2][itype] = force->numeric(FLERR, arg[ioffset + 5]);
	Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] = force->numeric(FLERR, arg[ioffset + 6]);
	Lookup[HEAT_CAPACITY][itype] = force->numeric(FLERR, arg[ioffset + 7]);

	Lookup[LAME_LAMBDA][itype] = Lookup[YOUNGS_MODULUS][itype] * Lookup[POISSON_RATIO][itype]
			/ ((1.0 + Lookup[POISSON_RATIO][itype] * (1.0 - 2.0 * Lookup[POISSON_RATIO][itype])));
	Lookup[SHEAR_MODULUS][itype] = Lookup[YOUNGS_MODULUS][itype] / (2.0 * (1.0 + Lookup[POISSON_RATIO][itype]));
	Lookup[M_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype];
	Lookup[SIGNAL_VELOCITY][itype] = sqrt(
			(Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype]) / Lookup[REFERENCE_DENSITY][itype]);
	Lookup[BULK_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype] / 3.0;

	if (comm->me == 0) {
		printf("\n material unspecific properties for SMD/TLSPH definition of particle type %d:\n", itype);
		printf("%60s : %g\n", "reference density", Lookup[REFERENCE_DENSITY][itype]);
		printf("%60s : %g\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
		printf("%60s : %g\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
		printf("%60s : %g\n", "linear viscosity coefficient", Lookup[VISCOSITY_Q1][itype]);
		printf("%60s : %g\n", "quadratic viscosity coefficient", Lookup[VISCOSITY_Q2][itype]);
		printf("%60s : %g\n", "hourglass control coefficient", Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]);
		printf("%60s : %g\n", "heat capacity [energy / (mass * temperature)]", Lookup[HEAT_CAPACITY][itype]);
		printf("%60s : %g\n", "Lame constant lambda", Lookup[LAME_LAMBDA][itype]);
		printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
		printf("%60s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
		printf("%60s : %g\n", "signal velocity", Lookup[SIGNAL_VELOCITY][itype]);

	}

	/*
	 * read following material cards
	 */

	//printf("next kwd is %s\n", arg[iNextKwd]);
	eos[itype] = EOS_NONE;
	strengthModel[itype] = STRENGTH_NONE;

	while (true) {
		if (strcmp(arg[iNextKwd], "*END") == 0) {
			if (comm->me == 0) {
				printf("found *END keyword");
				printf(
						"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n\n");
			}
			break;
		}

		/*
		 * Linear Elasticity model based on deformation gradient
		 */
		ioffset = iNextKwd;
		if (strcmp(arg[ioffset], "*LINEAR_DEFGRAD") == 0) {
			strengthModel[itype] = LINEAR_DEFGRAD;

			if (comm->me == 0) {
				printf("reading *LINEAR_DEFGRAD\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_DEFGRAD");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *LINEAR_DEFGRAD but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "Linear Elasticity model based on deformation gradient");
			}
		} else if (strcmp(arg[ioffset], "*LINEAR_STRENGTH") == 0) {

			/*
			 * Linear Elasticity strength only model based on strain rate
			 */

			strengthModel[itype] = LINEAR_STRENGTH;
			if (comm->me == 0) {
				printf("reading *LINEAR_STRENGTH\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_STRENGTH");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *LINEAR_STRENGTH but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("%60s\n", "Linear Elasticity strength based on strain rate");
			}
		} // end Linear Elasticity strength only model based on strain rate

		else if (strcmp(arg[ioffset], "*LINEAR_PLASTICITY") == 0) {

			/*
			 * Linear Elastic / perfectly plastic strength only model based on strain rate
			 */

			strengthModel[itype] = LINEAR_PLASTICITY;
			if (comm->me == 0) {
				printf("reading *LINEAR_PLASTICITY\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_PLASTICITY");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 2 + 1) {
				sprintf(str, "expected 2 arguments following *LINEAR_PLASTICITY but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[YIELD_STRESS][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[HARDENING_PARAMETER][itype] = force->numeric(FLERR, arg[ioffset + 2]);

			if (comm->me == 0) {
				printf("%60s\n", "Linear elastic / perfectly plastic strength based on strain rate");
				printf("%60s : %g\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
				printf("%60s : %g\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
				printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
				printf("%60s : %g\n", "constant yield stress", Lookup[YIELD_STRESS][itype]);
				printf("%60s : %g\n", "constant hardening parameter", Lookup[HARDENING_PARAMETER][itype]);
			}
		} // end Linear Elastic / perfectly plastic strength only model based on strain rate

		else if (strcmp(arg[ioffset], "*JOHNSON_COOK") == 0) {

			/*
			 * JOHNSON - COOK
			 */

			strengthModel[itype] = STRENGTH_JOHNSON_COOK;
			if (comm->me == 0) {
				printf("reading *JOHNSON_COOK\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *JOHNSON_COOK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 8 + 1) {
				sprintf(str, "expected 8 arguments following *JOHNSON_COOK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[JC_A][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[JC_B][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[JC_a][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			Lookup[JC_C][itype] = force->numeric(FLERR, arg[ioffset + 4]);
			Lookup[JC_epdot0][itype] = force->numeric(FLERR, arg[ioffset + 5]);
			Lookup[JC_T0][itype] = force->numeric(FLERR, arg[ioffset + 6]);
			Lookup[JC_Tmelt][itype] = force->numeric(FLERR, arg[ioffset + 7]);
			Lookup[JC_M][itype] = force->numeric(FLERR, arg[ioffset + 8]);

			if (comm->me == 0) {
				printf("%60s\n", "Johnson Cook material strength model");
				printf("%60s : %g\n", "A: initial yield stress", Lookup[JC_A][itype]);
				printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[JC_B][itype]);
				printf("%60s : %g\n", "a : exponent for plastic strain dependency", Lookup[JC_a][itype]);
				printf("%60s : %g\n", "C : proportionality factor for logarithmic plastic strain rate dependency",
						Lookup[JC_C][itype]);
				printf("%60s : %g\n", "epdot0 : dimensionality factor for plastic strain rate dependency",
						Lookup[JC_epdot0][itype]);
				printf("%60s : %g\n", "T0 : reference (room) temperature", Lookup[JC_T0][itype]);
				printf("%60s : %g\n", "Tmelt : melting temperature", Lookup[JC_Tmelt][itype]);
				printf("%60s : %g\n", "M : exponent for temperature dependency", Lookup[JC_M][itype]);
			}

		} else if (strcmp(arg[ioffset], "*EOS_NONE") == 0) {

			/*
			 * no eos
			 */

			eos[itype] = EOS_NONE;
			if (comm->me == 0) {
				printf("reading *EOS_NONE\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_NONE");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *EOS_NONE but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "no EOS selected");
			}

		} else if (strcmp(arg[ioffset], "*EOS_LINEAR") == 0) {

			/*
			 * linear eos
			 */

			eos[itype] = EOS_LINEAR;
			if (comm->me == 0) {
				printf("reading *EOS_LINEAR\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_LINEAR");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *EOS_LINEAR but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "linear EOS based on strain rate");
				printf("%60s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
			}
		} // end linear eos
		else if (strcmp(arg[ioffset], "*EOS_SHOCK") == 0) {

			/*
			 * shock eos
			 */

			eos[itype] = EOS_SHOCK;
			if (comm->me == 0) {
				printf("reading *EOS_SHOCK\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_SHOCK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 3 + 1) {
				sprintf(str, "expected 3 arguments following *EOS_SHOCK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[EOS_SHOCK_C0][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[EOS_SHOCK_S][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[EOS_SHOCK_GAMMA][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			if (comm->me == 0) {
				printf("\n%60s\n", "shock EOS based on strain rate");
				printf("%60s : %g\n", "reference speed of sound", Lookup[EOS_SHOCK_C0][itype]);
				printf("%60s : %g\n", "Hugoniot parameter S", Lookup[EOS_SHOCK_S][itype]);
				printf("%60s : %g\n", "Grueneisen Gamma", Lookup[EOS_SHOCK_GAMMA][itype]);
			}
		} // end shock eos

		else if (strcmp(arg[ioffset], "*EOS_POLYNOMIAL") == 0) {
			/*
			 * polynomial eos
			 */

			eos[itype] = EOS_POLYNOMIAL;
			if (comm->me == 0) {
				printf("reading *EOS_POLYNOMIAL\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_POLYNOMIAL");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 7 + 1) {
				sprintf(str, "expected 7 arguments following *EOS_POLYNOMIAL but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("eos_polynomial_c0", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
			matProp2[std::make_pair("eos_polynomial_c1", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
			matProp2[std::make_pair("eos_polynomial_c2", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
			matProp2[std::make_pair("eos_polynomial_c3", itype)] = force->numeric(FLERR, arg[ioffset + 4]);
			matProp2[std::make_pair("eos_polynomial_c4", itype)] = force->numeric(FLERR, arg[ioffset + 5]);
			matProp2[std::make_pair("eos_polynomial_c5", itype)] = force->numeric(FLERR, arg[ioffset + 6]);
			matProp2[std::make_pair("eos_polynomial_c6", itype)] = force->numeric(FLERR, arg[ioffset + 7]);
			if (comm->me == 0) {
				printf("\n%60s\n", "polynomial EOS based on strain rate");
				printf("%60s : %g\n", "parameter c0", SafeLookup("eos_polynomial_c0", itype));
				printf("%60s : %g\n", "parameter c1", SafeLookup("eos_polynomial_c1", itype));
				printf("%60s : %g\n", "parameter c2", SafeLookup("eos_polynomial_c2", itype));
				printf("%60s : %g\n", "parameter c3", SafeLookup("eos_polynomial_c3", itype));
				printf("%60s : %g\n", "parameter c4", SafeLookup("eos_polynomial_c4", itype));
				printf("%60s : %g\n", "parameter c5", SafeLookup("eos_polynomial_c5", itype));
				printf("%60s : %g\n", "parameter c6", SafeLookup("eos_polynomial_c6", itype));
			}
		} // end polynomial eos

		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PLASTIC_STRAIN") == 0) {

			/*
			 * maximum plastic strain failure criterion
			 */

			if (comm->me == 0) {
				printf("reading *FAILURE_MAX_PLASTIC_SRTRAIN\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PLASTIC_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PLASTIC_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("\n%60s\n", "maximum plastic strain failure criterion");
				printf("%60s : %g\n", "failure occurs when plastic strain reaches limit",
						Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]);
			}
		} // end maximum plastic strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRAIN") == 0) {

			/*
			 * maximum principal strain failure criterion
			 */
			if (comm->me == 0) {
				printf("reading *FAILURE_MAX_PRINCIPAL_STRAIN\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_max_principal_strain", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("\n%60s\n", "maximum principal strain failure criterion");
				printf("%60s : %g\n", "failure occurs when principal strain reaches limit",
						SafeLookup("failure_max_principal_strain", itype));
			}
		} // end maximum principal strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_JOHNSON_COOK") == 0) {

			if (comm->me == 0) {
				printf("reading *FAILURE_JOHNSON_COOK\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_JOHNSON_COOK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 5 + 1) {
				sprintf(str, "expected 5 arguments following *FAILURE_JOHNSON_COOK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_JC_d1", itype)] = force->numeric(FLERR, arg[ioffset + 1]);
			matProp2[std::make_pair("failure_JC_d2", itype)] = force->numeric(FLERR, arg[ioffset + 2]);
			matProp2[std::make_pair("failure_JC_d3", itype)] = force->numeric(FLERR, arg[ioffset + 3]);
			matProp2[std::make_pair("failure_JC_d4", itype)] = force->numeric(FLERR, arg[ioffset + 4]);
			matProp2[std::make_pair("failure_JC_epdot0", itype)] = force->numeric(FLERR, arg[ioffset + 5]);

			if (comm->me == 0) {
				printf("\n%60s\n", "Johnson-Cook failure criterion");
				printf("%60s : %g\n", "parameter d1", SafeLookup("failure_JC_d1", itype));
				printf("%60s : %g\n", "parameter d2", SafeLookup("failure_JC_d2", itype));
				printf("%60s : %g\n", "parameter d3", SafeLookup("failure_JC_d3", itype));
				printf("%60s : %g\n", "parameter d4", SafeLookup("failure_JC_d4", itype));
				printf("%60s : %g\n", "reference plastic strain rate", SafeLookup("failure_JC_epdot0", itype));
			}

		} else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRESS") == 0) {

			/*
			 * maximum principal stress failure criterion
			 */

			if (comm->me == 0) {
				printf("reading *FAILURE_MAX_PRINCIPAL_STRESS\n");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRESS");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRESS but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			matProp2[std::make_pair("failure_max_principal_stress", itype)] = force->numeric(FLERR, arg[ioffset + 1]);

			if (comm->me == 0) {
				printf("\n%60s\n", "maximum principal stress failure criterion");
				printf("%60s : %g\n", "failure occurs when principal stress reaches limit",
						SafeLookup("failure_max_principal_stress", itype));
			}
		} // end maximum principal stress failure criterion
		else {
			sprintf(str, "unknown *KEYWORD: %s", arg[ioffset]);
			error->all(FLERR, str);
		}

	}

	setflag[itype][itype] = 1;

}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairTlsph::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	if (force->newton == 1)
		error->all(FLERR, "Pair style tlsph requires newton off");

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
//printf("cutoff for pair pair tlsph = %f\n", cutoff);
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairTlsph::init_style() {
	int i;

	if (force->newton_pair == 1) {
		error->all(FLERR, "Pair style tlsph requires newton pair off");
	}

// request a granular neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;

// set maxrad_dynamic and maxrad_frozen for each type
// include future Fix pour particles as dynamic

	for (i = 1; i <= atom->ntypes; i++)
		onerad_dynamic[i] = onerad_frozen[i] = 0.0;

	double *radius = atom->radius;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	for (i = 0; i < nlocal; i++)
		onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);

	MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes,
	MPI_DOUBLE, MPI_MAX, world);
	MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes,
	MPI_DOUBLE, MPI_MAX, world);

	// if first init, create Fix needed for storing reference configuration neighbors

	int igroup = group->find("tlsph");
	if (igroup == -1)
		error->all(FLERR, "Pair style tlsph requires its particles to be part of a group named tlsph. This group does not exist.");

	if (fix_tlsph_reference_configuration == NULL) {
		char **fixarg = new char*[3];
		fixarg[0] = (char *) "SMD_TLSPH_NEIGHBORS";
		fixarg[1] = (char *) "tlsph";
		fixarg[2] = (char *) "SMD_TLSPH_NEIGHBORS";
		modify->add_fix(3, fixarg);
		delete[] fixarg;
		fix_tlsph_reference_configuration = (FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[modify->nfix - 1];
		fix_tlsph_reference_configuration->pair = this;
	}

	// find associated SMD_TLSPH_NEIGHBORS fix that must exist
	// could have changed locations in fix list since created

	ifix_tlsph = -1;
	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "SMD_TLSPH_NEIGHBORS") == 0)
			ifix_tlsph = i;
	if (ifix_tlsph == -1)
		error->all(FLERR, "Fix SMD_TLSPH_NEIGHBORS does not exist");

}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairTlsph::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairTlsph::memory_usage() {

	return 118 * nmax * sizeof(double);
}

/* ----------------------------------------------------------------------
 extract method to provide access to this class' data structures
 ------------------------------------------------------------------------- */

void *PairTlsph::extract(const char *str, int &i) {
//printf("in PairTlsph::extract\n");
	if (strcmp(str, "smd/tlsph/Fincr_ptr") == 0) {
		return (void *) Fincr;
	} else if (strcmp(str, "smd/tlsph/detF_ptr") == 0) {
		return (void *) detF;
	} else if (strcmp(str, "smd/tlsph/PK1_ptr") == 0) {
		return (void *) PK1;
	} else if (strcmp(str, "smd/tlsph/smoothVel_ptr") == 0) {
		return (void *) smoothVelDifference;
	} else if (strcmp(str, "smd/tlsph/numNeighsRefConfig_ptr") == 0) {
		return (void *) numNeighsRefConfig;
	} else if (strcmp(str, "smd/tlsph/stressTensor_ptr") == 0) {
		return (void *) CauchyStress;
	} else if (strcmp(str, "smd/tlsph/updateFlag_ptr") == 0) {
		return (void *) &updateFlag;
	} else if (strcmp(str, "smd/tlsph/strain_rate_ptr") == 0) {
		return (void *) D;
	} else if (strcmp(str, "smd/tlsph/hMin_ptr") == 0) {
		return (void *) &hMin;
	} else if (strcmp(str, "smd/tlsph/dtCFL_ptr") == 0) {
		return (void *) &dtCFL;
	} else if (strcmp(str, "smd/tlsph/dtRelative_ptr") == 0) {
		return (void *) &dtRelative;
	} else if (strcmp(str, "smd/tlsph/hourglass_error_ptr") == 0) {
		return (void *) hourglass_error;
	}

	return NULL;
}

/* ---------------------------------------------------------------------- */

int PairTlsph::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	int *mol = atom->molecule;
	double *damage = atom->damage;

//printf("in PairTlsph::pack_forward_comm\n");

	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = PK1[j](0, 0); // PK1 is not symmetric
		buf[m++] = PK1[j](0, 1);
		buf[m++] = PK1[j](0, 2);
		buf[m++] = PK1[j](1, 0);
		buf[m++] = PK1[j](1, 1);
		buf[m++] = PK1[j](1, 2);
		buf[m++] = PK1[j](2, 0);
		buf[m++] = PK1[j](2, 1);
		buf[m++] = PK1[j](2, 2); // 9

		buf[m++] = Fincr[j](0, 0); // Fincr is not symmetric
		buf[m++] = Fincr[j](0, 1);
		buf[m++] = Fincr[j](0, 2);
		buf[m++] = Fincr[j](1, 0);
		buf[m++] = Fincr[j](1, 1);
		buf[m++] = Fincr[j](1, 2);
		buf[m++] = Fincr[j](2, 0);
		buf[m++] = Fincr[j](2, 1);
		buf[m++] = Fincr[j](2, 2); // 9 + 9 = 18

		buf[m++] = shepardWeight[j]; // 19
		buf[m++] = mol[j];
		buf[m++] = damage[j];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairTlsph::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	int *mol = atom->molecule;
	double *damage = atom->damage;

//printf("in PairTlsph::unpack_forward_comm\n");

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {

		PK1[i](0, 0) = buf[m++]; // PK1 is not symmetric
		PK1[i](0, 1) = buf[m++];
		PK1[i](0, 2) = buf[m++];
		PK1[i](1, 0) = buf[m++];
		PK1[i](1, 1) = buf[m++];
		PK1[i](1, 2) = buf[m++];
		PK1[i](2, 0) = buf[m++];
		PK1[i](2, 1) = buf[m++];
		PK1[i](2, 2) = buf[m++];

		Fincr[i](0, 0) = buf[m++];
		Fincr[i](0, 1) = buf[m++];
		Fincr[i](0, 2) = buf[m++];
		Fincr[i](1, 0) = buf[m++];
		Fincr[i](1, 1) = buf[m++];
		Fincr[i](1, 2) = buf[m++];
		Fincr[i](2, 0) = buf[m++];
		Fincr[i](2, 1) = buf[m++];
		Fincr[i](2, 2) = buf[m++];

		shepardWeight[i] = buf[m++];
		mol[i] = static_cast<int>(buf[m++]);
		damage[i] = buf[m++];
	}
}

/*
 * Polar Decomposition via SVD, M = R * T
 * where R is a rotation and T a pure translation/stretch
 */

void PairTlsph::PolDec(Matrix3d &M, Matrix3d *R, Matrix3d *T) {

	JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV); // SVD(A) = U S V*
	Matrix3d S = svd.singularValues().asDiagonal();
	Matrix3d U = svd.matrixU();
	Matrix3d V = svd.matrixV();
	Matrix3d eye;
	eye.setIdentity();

// now do polar decomposition into M = R * T, where R is rotation
// and T is translation matrix
	*R = U * V.transpose();
	*T = V * S * V.transpose();

// now we check the polar decomposition
#ifdef TLSPH_DEBUG
	Matrix3d Mcheck = *R * R->transpose();
	Matrix3d Mdiff = Mcheck - eye;
	if (Mdiff.norm() > 1.0e-8) {
		printf("R is not orthogonal\n");
		cout << "Here is the Rotation matrix:" << endl << *R << endl;
	}

	if (fabs(R->determinant() - 1.0) > 1.0e-8) {
		printf("determinant of R=%f is not unity!\n", R->determinant());
	}

//    cout << "Here is the difference between M and its reconstruction from the polar decomp:" << endl << Mdiff << endl;
//    cout << "Here is the Rotation matrix:" << endl << *R << endl;
//    cout << "Here is the Translation Matrix:" << endl << *T << endl;
//    cout << "is U unitary? Her is U^T * U:" << endl << R->transpose()* *R << endl;
#endif
}

/*
 * deviator of a tensor
 */
Matrix3d PairTlsph::Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}

/*
 * Pseudo-inverse via SVD
 */

Matrix3d PairTlsph::pseudo_inverse_SVD(Matrix3d M) {

	JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV);

	Vector3d singularValuesInv;
	Vector3d singularValues = svd.singularValues();
	Matrix3d U = svd.matrixU();
	Matrix3d V = svd.matrixV();

//cout << "Here is the matrix V:" << endl << V * singularValues.asDiagonal() * U << endl;
//cout << "Its singular values are:" << endl << singularValues << endl;

	double pinvtoler = 1.0e-6;
	for (long row = 0; row < 3; row++) {
		if (singularValues(row) > pinvtoler) {
			singularValuesInv(row) = 1.0 / singularValues(row);
		} else {
			singularValuesInv(row) = 1.0;
		}
	}

	Matrix3d pInv;
	pInv = V * singularValuesInv.asDiagonal() * U.transpose();

	return pInv;
}

/*
 * test if two matrices are equal
 */
double PairTlsph::TestMatricesEqual(Matrix3d A, Matrix3d B, double eps) {
	Matrix3d diff;
	diff = A - B;
	double norm = diff.norm();
	if (norm > eps) {
		printf("Matrices A and B are not equal! The L2-norm difference is: %g\n", norm);
		cout << "Here is matrix A:" << endl << A << endl;
		cout << "Here is matrix B:" << endl << B << endl;
	}
	return norm;
}

/* ----------------------------------------------------------------------
 compute effective P-wave speed
 determined by longitudinal modulus
 ------------------------------------------------------------------------- */

void PairTlsph::effective_longitudinal_modulus(const int itype, const double dt, const double d_iso, const double p_rate,
		const Matrix3d d_dev, const Matrix3d sigma_dev_rate, const double damage, double &K_eff, double &mu_eff, double &M_eff) {
	double M0; // initial longitudinal modulus
	double shear_rate_sq;

//	if (damage >= 0.5) {
//		M_eff = Lookup[M_MODULUS][itype];
//		K_eff = Lookup[BULK_MODULUS][itype];
//		mu_eff = Lookup[SHEAR_MODULUS][itype];
//		return;
//	}

	M0 = Lookup[M_MODULUS][itype];

	if (dt * d_iso > 1.0e-6) {
		K_eff = p_rate / d_iso;
		if (K_eff < 0.0) { // it is possible for K_eff to become negative due to strain softening
//			if (damage == 0.0) {
//				error->one(FLERR, "computed a negative effective bulk modulus but particle is not damaged.");
//			}
			K_eff = Lookup[BULK_MODULUS][itype];
		}
	} else {
		K_eff = Lookup[BULK_MODULUS][itype];
	}

	if (domain->dimension == 3) {
		// Calculate 2 mu by looking at ratio shear stress / shear strain. Use numerical softening to avoid divide-by-zero.
		mu_eff = 0.5
				* (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16) + sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)
						+ sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16));

		// Calculate magnitude of deviatoric strain rate. This is used for deciding if shear modulus should be computed from current rate or be taken as the initial value.
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2);
	} else {
		mu_eff = 0.5 * (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16));
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1);
	}

	if (dt * dt * shear_rate_sq < 1.0e-8) {
		mu_eff = Lookup[SHEAR_MODULUS][itype];
	}

	if (mu_eff < Lookup[SHEAR_MODULUS][itype]) { // it is possible for mu_eff to become negative due to strain softening
//		if (damage == 0.0) {
//			printf("mu_eff = %f, tau=%f, gamma=%f\n", mu_eff, sigma_dev_rate(0, 1), d_dev(0, 1));
//			error->message(FLERR, "computed a negative effective shear modulus but particle is not damaged.");
//		}
		mu_eff = Lookup[SHEAR_MODULUS][itype];
	}

	//mu_eff = Lookup[SHEAR_MODULUS][itype];

	if (K_eff < 0.0) {
		printf("K_eff = %f, p_rate=%f, vol_rate=%f\n", K_eff, p_rate, d_iso);
	}

	if (mu_eff < 0.0) {
		printf("mu_eff = %f, tau=%f, gamma=%f\n", mu_eff, sigma_dev_rate(0, 1), d_dev(0, 1));
		error->one(FLERR, "");
	}

	M_eff = (K_eff + 4.0 * mu_eff / 3.0); // effective dilational modulus, see Pronto 2d eqn 3.4.8

	if (M_eff < M0) { // do not allow effective dilatational modulus to decrease beyond its initial value
		M_eff = M0;
	}
}

double PairTlsph::SafeLookup(std::string str, int itype) {
	cout << "string passed to lookup: " << str.c_str() << endl;
	error->all(FLERR, "");
	char msg[128];
	if (matProp2.count(std::make_pair(str, itype)) == 1) {
		cout << "returning look up value %d " << matProp2[std::make_pair(str, itype)] << endl;
		return matProp2[std::make_pair(str, itype)];
	} else {
		sprintf(msg, "failed to lookup indentifier [%s] for particle type %d", str.c_str(), itype);
		error->all(FLERR, msg);
	}
	return 1.0;
}

bool PairTlsph::CheckKeywordPresent(std::string str, int itype) {
	int count = matProp2.count(std::make_pair(str, itype));
	if (count == 0) {
		return false;
	} else if (count == 1) {
		return true;
	} else {
		char msg[128];
		cout << "keyword: " << str << endl;
		sprintf(msg, "ambiguous count for keyword: %d times present\n", count);
		error->all(FLERR, msg);
	}
	return false;
}

/* ----------------------------------------------------------------------
 Limit eigenvalues of a matrix to upper and lower bounds.
 ------------------------------------------------------------------------- */

Matrix3d PairTlsph::LimitEigenvalues(Matrix3d S, double limitEigenvalue) {

	/*
	 * compute Eigenvalues of matrix S
	 */
	SelfAdjointEigenSolver<Matrix3d> es;
	es.compute(S);

	double max_eigenvalue = es.eigenvalues().maxCoeff();
	double min_eigenvalue = es.eigenvalues().minCoeff();
	double amax_eigenvalue = fabs(max_eigenvalue);
	double amin_eigenvalue = fabs(min_eigenvalue);

	if ((amax_eigenvalue > limitEigenvalue) || (amin_eigenvalue > limitEigenvalue)) {
		if (amax_eigenvalue > amin_eigenvalue) { // need to scale with max_eigenvalue
			double scale = amax_eigenvalue / limitEigenvalue;
			Matrix3d V = es.eigenvectors();
			Matrix3d S_diag = V.inverse() * S * V; // diagonalized input matrix
			S_diag /= scale;
			Matrix3d S_scaled = V * S_diag * V.inverse(); // undiagonalize matrix
			return S_scaled;
		} else { // need to scale using min_eigenvalue
			double scale = amin_eigenvalue / limitEigenvalue;
			Matrix3d V = es.eigenvectors();
			Matrix3d S_diag = V.inverse() * S * V; // diagonalized input matrix
			S_diag /= scale;
			Matrix3d S_scaled = V * S_diag * V.inverse(); // undiagonalize matrix
			return S_scaled;
		}
	} else { // limiting does not apply
		return S;
	}
}

/* ----------------------------------------------------------------------
 compute pressure. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputePressure(const int i, const double pInitial, const double d_iso, double &pFinal, double &p_rate) {
	double *rmass = atom->rmass;
	double *vfrac = atom->vfrac;
	double *e = atom->e;
	int *type = atom->type;
	double dt = update->dt;

	int itype;

	itype = type[i];

	double mass_specific_energy = e[i] / rmass[i]; // energy per unit mass
	double rho = rmass[i] / (detF[i] * vfrac[i]); // current mass density
	double vol_specific_energy = mass_specific_energy * rho; // energy per current volume

	switch (eos[itype]) {
	case EOS_LINEAR:
		LinearEOS(Lookup[BULK_MODULUS][itype], pInitial, d_iso, dt, pFinal, p_rate);
		break;
	case EOS_NONE:
		pFinal = 0.0;
		p_rate = 0.0;
		break;
	case EOS_SHOCK:
		//  rho,  rho0,  e,  e0,  c0,  S,  Gamma,  pInitial,  dt,  &pFinal,  &p_rate);
		ShockEOS(rho, Lookup[REFERENCE_DENSITY][itype], mass_specific_energy, 0.0, Lookup[EOS_SHOCK_C0][itype],
				Lookup[EOS_SHOCK_S][itype], Lookup[EOS_SHOCK_GAMMA][itype], pInitial, dt, pFinal, p_rate);
		break;
	case EOS_POLYNOMIAL:
		polynomialEOS(rho, Lookup[REFERENCE_DENSITY][itype], vol_specific_energy, SafeLookup("eos_polynomial_c0", itype),
				SafeLookup("eos_polynomial_c1", itype), SafeLookup("eos_polynomial_c2", itype),
				SafeLookup("eos_polynomial_c3", itype), SafeLookup("eos_polynomial_c4", itype),
				SafeLookup("eos_polynomial_c5", itype), SafeLookup("eos_polynomial_c6", itype), pInitial, dt, pFinal, p_rate);

		break;
	default:
		error->one(FLERR, "unknown EOS.");
		break;
	}
}

/* ----------------------------------------------------------------------
 Compute stress deviator. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeStressDeviator(const int i, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev,
		Matrix3d &sigma_dev_rate, double &plastic_strain_increment) {
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *type = atom->type;
	double *rmass = atom->rmass;
	//double *vfrac = atom->vfrac;
	double *e = atom->e;
	double dt = update->dt;
	double yieldStress;
	int itype;

	double mass_specific_energy = e[i] / rmass[i]; // energy per unit mass
	plastic_strain_increment = 0.0;
	itype = type[i];

	switch (strengthModel[itype]) {
	case LINEAR_STRENGTH:
		//printf("mu0 is %f\n", mu0[itype]);
		//LinearStrength(Lookup[SHEAR_MODULUS][itype], sigmaInitial_dev, d[i], dt, &sigmaFinal_dev, &sigma_dev_rate);

		sigma_dev_rate = 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
		sigmaFinal_dev = sigmaInitial_dev + dt * sigma_dev_rate;

		break;
	case LINEAR_DEFGRAD:
		//LinearStrengthDefgrad(Lookup[LAME_LAMBDA][itype], Lookup[SHEAR_MODULUS][itype], Fincr[i], &sigmaFinal_dev);
		//eff_plastic_strain[i] = 0.0;
		//p_rate = pInitial - sigmaFinal_dev.trace() / 3.0;
		//sigma_dev_rate = sigmaInitial_dev - Deviator(sigmaFinal_dev);
		error->one(FLERR, "LINEAR_DEFGRAD is only for debugging purposes and currently deactivated.");
		R[i].setIdentity();
		break;
	case LINEAR_PLASTICITY:

		yieldStress = Lookup[YIELD_STRESS][itype] + Lookup[HARDENING_PARAMETER][itype] * eff_plastic_strain[i];
		LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev,
				sigma_dev_rate, plastic_strain_increment);
		break;
	case STRENGTH_JOHNSON_COOK:
		JohnsonCookStrength(Lookup[SHEAR_MODULUS][itype], Lookup[HEAT_CAPACITY][itype], mass_specific_energy, Lookup[JC_A][itype],
				Lookup[JC_B][itype], Lookup[JC_a][itype], Lookup[JC_C][itype], Lookup[JC_epdot0][itype], Lookup[JC_T0][itype],
				Lookup[JC_Tmelt][itype], Lookup[JC_M][itype], dt, eff_plastic_strain[i], eff_plastic_strain_rate[i],
				sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment);
		break;
	case STRENGTH_NONE:
		sigmaFinal_dev.setZero();
		sigma_dev_rate.setZero();
		break;
	default:
		error->one(FLERR, "unknown strength model.");
		break;
	}

}

/* ----------------------------------------------------------------------
 Compute damage. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeDamage(const int i, const Matrix3d strain, const double pFinal, const Matrix3d sigmaFinal,
		const Matrix3d sigmaFinal_dev, Matrix3d &sigma_damaged, double &damage_increment) {
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *type = atom->type;
	double *radius = atom->radius;
	double dt = update->dt;
	int itype = type[i];
	bool damage_flag = false;

	if (CheckKeywordPresent("failure_max_principal_stress", itype)) {
		/*
		 * maximum stress failure criterion:
		 */
		damage_flag = IsotropicMaxStressDamage(strain, SafeLookup("failure_max_principal_stress", itype));
	}
	if (CheckKeywordPresent("failure_max_principal_strain", itype)) {
		/*
		 * maximum strain failure criterion:
		 */
		damage_flag = IsotropicMaxStrainDamage(strain, SafeLookup("failure_max_principal_strain", itype));
	}

	if (Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype] > 0.0) {

		if (eff_plastic_strain[i] >= 0.99 * Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]) {
			damage_flag = true;
		}

	}

	if (CheckKeywordPresent("failure_JC_d1", itype)) {

		double jc_failure_strain = JohnsonCookFailureStrain(pFinal, sigmaFinal_dev, SafeLookup("failure_JC_d1", itype),
				SafeLookup("failure_JC_d2", itype), SafeLookup("failure_JC_d3", itype), SafeLookup("failure_JC_d4", itype),
				SafeLookup("failure_JC_epdot0", itype), eff_plastic_strain_rate[i]);

		//cout << "plastic strain increment is " << plastic_strain_increment << "  jc fs is " << jc_failure_strain << endl;
		if (eff_plastic_strain[i] / jc_failure_strain > 1.0) {
			damage_flag = true;
		}
	}

	/*
	 * Damage does not happen instantaneously but rather takes time which signal_velocity takes to go over a distance h.
	 * Therefore compute a damage increment which is time-integrated in AssembleStress().
	 */

	if (damage_flag == true) {
		damage_increment = dt * Lookup[SIGNAL_VELOCITY][itype] / (100.0 * radius[i]);
	} else {
		damage_increment = 0.0;
	}
}

void PairTlsph::spiky_kernel_and_derivative(const double h, const double r, double &wf, double &wfd) {

	/*
	 * Spiky kernel
	 */

	if (r >= h) {
		//printf("r=%f > h=%f in Spiky kernel\n", r, h);
		wf = wfd = 0.0;
		return;
	}
	double hr = h - r;		// [m]
	if (domain->dimension == 2) {
		double n = 0.3141592654e0 * h * h * h * h * h; // [m^5]
		wfd = -3.0e0 * hr * hr / n; // [m*m/m^5] = [1/m^3] ==> correct for dW/dr in 2D
		wf = -0.333333333333e0 * hr * wfd; // [m/m^3] ==> [1/m^2] correct for W in 2D
	} else {
		wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h); // [1/m^4] ==> correct for dW/dr in 3D
		wf = -0.333333333333e0 * hr * wfd; // [m/m^4] ==> [1/m^3] correct for W in 3D
	}

}
