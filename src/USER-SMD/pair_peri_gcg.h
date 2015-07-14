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

PairStyle(smd/peri_pmb,PairPeriGCG)

#else

//ss

#ifndef LMP_PAIR_PERI_GCG_H
#define LMP_PAIR_PERI_GCG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPeriGCG : public Pair {
 public:
  PairPeriGCG(class LAMMPS *);
  virtual ~PairPeriGCG();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();
  void init_list(int, class NeighList *);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *) {}
  void read_restart_settings(FILE *) {}
  virtual double memory_usage();

 protected:
  int ifix_peri;
  double **bulkmodulus;
  double **syield, **smax, **alpha, **G0, **c0;
  double **rho0;
  double cutoff_global;

  void allocate();

  int nBroken; // number of broken bonds
};

}

#endif
#endif
