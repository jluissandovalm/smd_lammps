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

#ifdef PAIR_CLASS

PairStyle(tlsph,PairTlsph)

#else

#ifndef LMP_TLSPH_NEW_H
#define LMP_TLSPH_NEW_H

#include "pair.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <map>

using namespace std;
using namespace Eigen;
namespace LAMMPS_NS {

class PairTlsph: public Pair {
public:

	PairTlsph(class LAMMPS *);
	virtual ~PairTlsph();
	virtual void compute(int, int);
	void settings(int, char **);
	void coeff(int, char **);
	double init_one(int, int);
	void init_style();
	void init_list(int, class NeighList *);
	void write_restart_settings(FILE *) {
	}
	void read_restart_settings(FILE *) {
	}
	virtual double memory_usage();
	void compute_shape_matrix(void);
	void material_model(void);
	void *extract(const char *, int &);
	int pack_forward_comm(int, int *, double *, int, int *);
	void unpack_forward_comm(int, int, double *);
	void AssembleStress();

	void PreCompute();
	void ComputeForces(int eflag, int vflag);
	void effective_longitudinal_modulus(const int itype, const double dt, const double d_iso, const double p_rate,
			const Matrix3d d_dev, const Matrix3d sigma_dev_rate, const double damage, double &K_eff, double &mu_eff, double &M_eff);

	void ComputePressure(const int i, const double pInitial, const double d_iso, double &pFinal, double &p_rate);
	void ComputeStressDeviator(const int i, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev,
			Matrix3d &sigma_dev_rate, double &plastic_strain_increment);
	void ComputeDamage(const int i, const Matrix3d strain, const double pFinal, const Matrix3d sigmaFinal,
			const Matrix3d sigmaFinal_dev, const double plastic_strain_increment, Matrix3d &sigma_damaged);
	void spiky_kernel_and_derivative(const double h, const double r, double &wf, double &wfd);

protected:
	void allocate();
	char *suffix;

	/*
	 * per-type arrays
	 */
	int *strengthModel, *eos;
	int *failureModel;
	double *onerad_dynamic, *onerad_frozen, *maxrad_dynamic, *maxrad_frozen;

	/*
	 * per atom arrays
	 */
	Matrix3d *K, *PK1, *Fdot, *Fincr;
	Matrix3d *d; // unrotated rate-of-deformation tensor
	Matrix3d *R; // rotation matrix
	Matrix3d *FincrInv;
	Matrix3d *D, *W; // strain rate and spin tensor
	Vector3d *smoothVelDifference;
	Matrix3d *CauchyStress;
	double *detF, *shepardWeight, *particle_dt;
	double *hourglass_error;
	int *numNeighsRefConfig;

	int nmax; // max number of atoms on this proc
	double hMin; // minimum kernel radius for two particles
	double dtCFL;
	double dtRelative; // relative velocity of two particles, divided by sound speed
	int updateFlag;
	double update_threshold; // updateFlage is set to one if the relative displacement of a pair exceeds update_threshold
	double cut_comm;

	enum {
		UPDATE_NONE = 5000
	};

	enum {
		LINEAR_DEFGRAD = 0,
		LINEAR_STRENGTH = 1,
		LINEAR_PLASTICITY = 2,
		STRENGTH_JOHNSON_COOK = 3,
		STRENGTH_NONE = 4,
		EOS_LINEAR = 5,
		EOS_SHOCK = 6,
		EOS_POLYNOMIAL = 7,
		EOS_NONE = 8,
		UPDATE_CONSTANT_THRESHOLD = 9,
		UPDATE_PAIRWISE_RATIO = 10,
		REFERENCE_DENSITY = 11,
		YOUNGS_MODULUS = 12,
		POISSON_RATIO = 13,
		HOURGLASS_CONTROL_AMPLITUDE = 14,
		HEAT_CAPACITY = 15,
		LAME_LAMBDA = 16,
		SHEAR_MODULUS = 17,
		M_MODULUS = 18,
		SIGNAL_VELOCITY = 19,
		BULK_MODULUS = 20,
		VISCOSITY_Q1 = 21,
		VISCOSITY_Q2 = 22,
		YIELD_STRESS = 23,
		FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD = 24,
		JC_A = 25,
		JC_B = 26,
		JC_a = 27,
		JC_C = 28,
		JC_epdot0 = 29,
		JC_T0 = 30,
		JC_Tmelt = 31,
		JC_M = 32,
		EOS_SHOCK_C0 = 33,
		EOS_SHOCK_S = 34,
		EOS_SHOCK_GAMMA = 35,
		HARDENING_PARAMETER = 36,
		FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD = 37,
		FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD = 38,
		MAX_KEY_VALUE = 39
	};

	// enumeration of failure / damage models
	enum {
		FAILURE_NONE = 4000,
		FAILURE_MAX_PRINCIPAL_STRAIN = 4001,
		FAILURE_MAX_PRINCIPAL_STRESS = 4002,
		FAILURE_MAX_PLASTIC_STRAIN = 4003,
		FAILURE_JOHNSON_COOK = 4004
	};

	// C++ std dictionary to hold material model settings per particle type
	typedef std::map<std::pair<std::string, int>, double> Dict;
	Dict matProp2;
	//typedef Dict::const_iterator It;

	int ifix_tlsph;
	int update_method;

	class FixSMD_TLSPH_ReferenceConfiguration *fix_tlsph_reference_configuration;

private:
	double SafeLookup(std::string str, int itype);
	bool CheckKeywordPresent(std::string str, int itype);
	double **Lookup; // holds per-type material parameters for the quantities defined in enum statement above.
	bool first; // if first is true, do not perform any computations, beacuse reference configuration is not ready yet.
};

}

#endif
#endif

/*
 * materialCoeffs array for EOS parameters:
 * 1: rho0
 *
 *
 * materialCoeffs array for strength parameters:
 *
 * Common
 * 10: maximum strain threshold for damage model
 * 11: maximum stress threshold for damage model
 *
 * Linear Plasticity model:
 * 12: plastic yield stress
 *
 *
 * Blei: rho = 11.34e-6, c0=2000, s=1.46, Gamma=2.77
 * Stahl 1403: rho = 7.86e-3, c=4569, s=1.49, Gamma=2.17
 */

