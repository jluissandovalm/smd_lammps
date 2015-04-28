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

#ifndef SMD_MATH_H_
#define SMD_MATH_H_

namespace SMD_Math {
static inline void LimitDoubleMagnitude(double &x, const double limit) {
	/*
	 * if |x| exceeds limit, set x to limit with the sign of x
	 */
	if (fabs(x) > limit) { // limit delVdotDelR to a fraction of speed of sound
		x = limit * copysign(1, x);
	}
}

}

#endif /* SMD_MATH_H_ */
