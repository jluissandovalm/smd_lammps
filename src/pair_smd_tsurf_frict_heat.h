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

PairStyle(smd/tsurf_frict_heat,PairTSurfFrictHeat)

#else

#ifndef LMP_SMD_TSURF_FRIC_HEAT_H
#define LMP_SMD_TSURF_FRIC_HEAT_H

#include "pair.h"
#include <Eigen/Eigen>
using namespace Eigen;


namespace LAMMPS_NS {

class PairTSurfFrictHeat : public Pair {
 public:
  PairTSurfFrictHeat(class LAMMPS *);
  virtual ~PairTSurfFrictHeat();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();
  void init_list(int, class NeighList *);
  virtual double memory_usage();
  void PointTriangleDistance(const Vector3d P, const Vector3d TRI1, const Vector3d TRI2, const Vector3d TRI3,
  		Vector3d &CP, double &dist);
  double clamp(const double a, const double min, const double max);
  void *extract(const char *, int &);

 protected:
  double **bulkmodulus; //
  double **kn;          //
                        //
  double **friCoeff;    //
  double **wallTemp;    //
  double **heatCap;     //
  double **diffusiv;    // per-type arrays read in from pair_coeff line

  double *onerad_dynamic,*onerad_frozen;
  double *maxrad_dynamic,*maxrad_frozen;

  double scale;
  double stable_time_increment; // stable time step size

  void allocate();

 private:
  bool friction, heat;

};

}

#endif
#endif

