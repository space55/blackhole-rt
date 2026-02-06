// kerr-image.c - compute an image of a Kerr black hole
// Written by David A. Madore (<URL: http://www.madore.org/~david/ >)
// This version: 2011-03-22.

/* This program is in the Public Domain: you can do whatever you wish
 * with it (but it also comes with no guarantee whatsoever).  However,
 * I would appreciate being credited if significant parts of it, or
 * images produced with it, are being used somewhere. */

/* For more information (and examples of output), visit
 *     <URL: http://www.madore.org/~david/math/kerr.html >,
 * and <URL: http://www.madore.org/~david/programs/#prog_kerr-image >. */

/* Concise usage instructions:

 * Under Linux, compile with something like
   gcc -o kerr-image kerr-image.c -O6 -Wall -std=c99 -pedantic \
     -Wextra -Wno-unused-result -lm
 * but this is standard C99, so it should probably compile cleanly
 * under just about any OS.

 * This program reads 22 numbers from standard input, separated by
 * whitespace.  Their meaning is as follows:
 * - first number is a float, and is ignored (typically used to store
 *   observer's proper time when making animations),
 * - second number is the integer 0, 1 or 2, to indicate whether all
 *   the following coordinates are given in ingoing Kerr (0), outgoing
 *   Kerr (1) or Boyer-Lindquist (2) coordinates, (note that for
 *   ingoing and outgoing Kerr coordinates, the t coordinate is V-r or
 *   V+r, not the null coordinate V),
 * - next four numbers are the coordinates of the observer's current
 *   position, in the order: r, cos(theta), t and phi (note the order,
 *   and note that it's cos(theta) and not theta itself which is
 *   given),
 * - next four numbers are the coordinates of the observer's current
 *   velocity, in the same order and convention (i.e., dr/ds,
 *   -sin(theta)*dtheta/ds, dt/ds and dphi/ds),
 * - next four numbers are the components of the vector indicating the
 *   direction in which the observer is looking (forward-pointing
 *   vector), again with the same order and convention,
 * - next four numbers are the components of the vector indicating the
 *   direction which the observer thinks is "right" (right-pointing
 *   vector), again with the same order and convention,
 * - final four numbers are the components of the vector indicating
 *   the direction which the observer thinks is "up" (up-pointing
 *   vector), again with the same order and convention.

 * The four vectors (velocity and forward-, right- and up- pointing)
 * will be orthonormalized by the Gram-Schmidt algorithm prior to any
 * processing by the program (and their values before and after this
 * orthonormalization will be printed out to standard error).  Of
 * course, the first vector should be timelike and the next three
 * should be spacelike (or you will get no meaningful output).

 * Here is an example possible set of input values, producing a view
 * which is not too uninteresting:
   0
   2
   10 0.5 0 0
   0 0 1 0
   -1 0 0 0
   0 0 0 1
   0 1 0 0
 * This will compute the view of the black hole by an observer sitting
 * motionless with respect to distant stars at a distance of 10 units
 * (5 Schwarzschild radii) at a latitude of 30 degrees (theta=pi/3),
 * and looking straight down into the black hole.

 * The program then produces a ppm file on standard output, whose
 * number of lines and columns is given by NBLINS and NBCOLS defined
 * at compile time.  By default, the viewing angle is set to about 100
 * degrees vertically and even more horizontally (it is assumed that
 * pixels are square), but a zoom factor can be introduced using the
 * variable ZOOM_FACTOR at compilation.  To produce partial images,
 * specify a range of lines on the command line (e.g. "kerr image 120
 * 240" to compute lines from 120 to 239 inclusive), and concatenate
 * the resulting output files (only the concatenation will be a
 * well-formed ppm file, not each individual output); the default is
 * to produce a full image of the size specified at compile time.

 * The black hole parameters are specied by BLACKHOLE_MASS and
 * BLACKHOLE_ANGMOM_PER_MASS at compile time (by default 1 and 0.8).
 * It is advised to leave BLACKHOLE_MASS untouched, since everything
 * would only change by a scale factor anyway.

 * The space is orned as followed: a blue sphere is placed outside the
 * black hole at some distance, a purple sphere is placed in negative
 * space (i.e., beyond the singularity cut), and the outer and inner
 * horizons are various shades of red and green (red/orange/brown for
 * outer, green for inner; lighter shades are white hole horizons,
 * darker shades are black hole horizons).  All these spheres are
 * checkered in an identical way, with twenty-four longitudinal
 * stripes and twelve latitudinal (or polar) stripes, consistent with
 * the black hole's axis.  (The longitudinal stripes on the horizons
 * rotate with the black hole; the longitudinal stripes on the distant
 * and negative spheres do not rotate with respect to distant stars.)
 * The radii of the distant and negative spheres can be changed with
 * the FARSPHERE_RADIUS and NEGSPHERE_RADIUS parameters at compile
 * time (to change the way things are checkered, dive into the
 * checkered() and main() functions).

 * The way this works is that geodesics for each light ray arriving at
 * the observer are integrated, partially using the first integrals of
 * motion, for every pixel in the image.  ("Partially" here means that
 * the equations on r and cos(theta) are kept as second-order because
 * this avoids a degeneracy when the derivative tends to zero, but the
 * value of the derivative is regularly recomputed.)  The geodesic
 * integration part of this program does NOT assume that the geodesic
 * is lightlike (as it is here), so a trivial modification of it can
 * be used to compute the path of timelike or spacelike geodesics.

 * Note: this program does not handle correctly the region where two
 * horizons cross (where neither ingoing nor outgoing Kerr coordinates
 * are adequate).  This will result in some "snow" around such
 * regions.  This is a numerical artefact.  Also, rays coming very
 * close to the axis (or, a fortiori, interesting it) will cause
 * problems.

 * Note: this program does not attempt to do any anti-aliasing on the
 * resulting image.  Do do this, compute a larger image, and downscale
 * it.

 * For more compilation-time settings, see below.

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef NBLINS				// Number of lines in image
#define NBLINS 240
#endif

#ifndef NBCOLS				// Number of columns in image
#define NBCOLS 320
#endif

#ifndef DO_USELESS			// Keep useless variables in integration
#define DO_USELESS 0
#endif

#ifndef MAX_NBSTEPS			// Iteration steps before giving up
#define MAX_NBSTEPS 100000
#endif

#ifndef MAX_PROPERTIME			// Max light ray time before giving up
#define MAX_PROPERTIME 200
#endif

#ifndef BASIC_STEP			// Basic integration step
					// (decrease for possibly better image)
#define BASIC_STEP 0.001
#endif

#ifndef ZOOM_FACTOR			// How much to zoom image around center
#define ZOOM_FACTOR 1
#endif

#ifndef FARSPHERE_RADIUS		// Radius of blue sphere
#define FARSPHERE_RADIUS 20.
#endif

#ifndef NEGSPHERE_RADIUS		// Radius of purple (negative) sphere
#define NEGSPHERE_RADIUS -20.
#endif

#ifndef BLACKHOLE_MASS			// Black hole mass (don't touch this)
#define BLACKHOLE_MASS 1.
#endif

#ifndef BLACKHOLE_ANGMOM_PER_MASS	// Black hole a parameter
#define BLACKHOLE_ANGMOM_PER_MASS 0.8
#endif

#ifndef M_PI // DO NOT TOUCH THIS!
#define M_PI 3.1415926535897932384626433832795029
#endif

static inline double
sign (char neg)
{
  return (neg?-1.:1.);
}

static inline double
square (double t)
{
  return t*t;
}

struct blackhole_s {
  double M;   // Black hole mass
  double a;   // Black hole rotation (angular momentum per unit mass)
  double M2;  // M*M
  double a2, a3, a4, a5, a6;  // Powers of a
  double sqrtM2ma2;  // sqrt(M^2-a^2)
};

void
init_blackhole (double M, double a, struct blackhole_s *blk)
// Initialize stupid constants which depend only on black hole itself (M, a).
{
  blk->M = M;
  blk->a = a;
  blk->M2 = M * M;
  blk->a2 = a * a;
  blk->a3 = (blk->a2) * a;
  blk->a4 = (blk->a2) * (blk->a2);
  blk->a5 = (blk->a3) * (blk->a2);
  blk->a6 = (blk->a3) * (blk->a3);
  blk->sqrtM2ma2 = sqrt((blk->M2) - (blk->a2));
}

struct geodesic_s {
  const struct blackhole_s *blk;
  // The first integrals of motion:
  double masssq;  // Mass square msq (+1 for timelike, 0 for lightlike, -1 for spacelike)
  double energy;  // Energy E (opposite of dot prod with (d/dt))
  double angmom;  // Angular momentum L around z axis (dot prod with (d/dphi))
  double carter;  // Carter's constant K
  // Two expressions involving these:
  double expr1;   // -K + a^2*(2*E^2-msq) - 2*a*L*E
  double expr2;   // 2*(E^2 - msq)
};

void
init_geodesic (const struct blackhole_s *blk, double masssq, double energy,
	       double angmom, double carter,
	       struct geodesic_s *geo)
// Initialize the structure describing a geodesic.
{
  geo->blk = blk;
  geo->masssq = masssq;
  geo->energy = energy;
  geo->angmom = angmom;
  geo->carter = carter;
  geo->expr1 = -carter + blk->a2*(2*square(energy)-masssq) - 2*blk->a*angmom*energy;
  geo->expr2 = 2*(square(energy)-masssq);
}

struct state_s {
  const struct blackhole_s *blk;
  const struct geodesic_s *geo;
  double time;  // Proper time
  // Coordinate system being used:
  char coords;  // 0 = ingoing, 1 = outgoing, 2 = Boyer-Lindquist
  union variables_u {
    struct variables_s {
      double r;
      double cth; // cos(theta)
      double rdot;  // dr/ds (we use a second order eqn for r)
      double cthdot; // (d/ds)(cos(theta)) (ditto)
#if DO_USELESS
      double l; // l is sqrt(r^2+a^2)*sin(theta)
#endif
      double t; // time coordinate (depends on value of coords)
      double phi; // phi coordinate (ditto)
#if DO_USELESS
      double cphsth; // cos(phi)*sin(theta)
      double sphsth; // sin(phi)*sin(theta)
#endif
    } n;
#define NBVARS ((int)(sizeof(struct variables_s)/sizeof(double)))
    double t[NBVARS];  // The same variables as an array, for use in runge_kutta()
  } v;
  int nbsteps;
};

void
change_coords (struct state_s *sta, char newcoords)
// Keep the same state, but change to a new coordinate system.
{
  char oldcoords = sta->coords;
  if ( oldcoords == newcoords )
    return;
  const struct blackhole_s *blk = sta->blk;
  double r = sta->v.n.r;
  double horlog = log(fabs((r-(blk->M+blk->sqrtM2ma2))/(r-(blk->M-blk->sqrtM2ma2))));
  double delta = square(r) - 2*blk->M*r + blk->a2;
  double tcorr = 2*(blk->M2/blk->sqrtM2ma2)*horlog + 2*blk->M*log(fabs(delta));
  double phicorr = (blk->a/blk->sqrtM2ma2)*horlog;
  double e = (newcoords==0?0.5:newcoords==1?-0.5:0.)
    - (oldcoords==0?0.5:oldcoords==1?-0.5:0.);
  sta->coords = newcoords;
  sta->v.n.t += e*tcorr;
  sta->v.n.phi += e*phicorr;
#if DO_USELESS
  double newcphsth = sta->v.n.cphsth*cos(e*phicorr) - sta->v.n.sphsth*sin(e*phicorr);
  double newsphsth = sta->v.n.sphsth*cos(e*phicorr) + sta->v.n.cphsth*sin(e*phicorr);
  sta->v.n.cphsth = newcphsth;
  sta->v.n.sphsth = newsphsth;
#endif
}

void
change_coords_to_best (struct state_s *sta)
// Change to the best coordinate system for integration.
{
  double p = (square(sta->v.n.r)+sta->blk->a2)*sta->geo->energy
    - sta->blk->a*sta->geo->angmom;
  char newcoords = (sta->v.n.rdot > 0) ^ (p < 0);
  change_coords (sta, newcoords);
}

void
init_state (const struct geodesic_s *geo, double time, double r, double cth,
	    char rdot_sign, char cthdot_sign,
	    char coords, double t, double phi,
	    struct state_s *sta)
// Initialize state structure.
{
  const struct blackhole_s *blk = geo->blk;
  sta->blk = blk;
  sta->geo = geo;
  sta->time = time;
  sta->v.n.r = r;
  sta->v.n.cth = cth;
  double rhosq = square(r) + blk->a2*square(cth);
  sta->v.n.rdot = sign(rdot_sign)*sqrt((-geo->masssq*square(r)-geo->carter)*(square(r)-2*blk->M*r+blk->a2) + square((square(r)+blk->a2)*geo->energy - blk->a*geo->angmom))/rhosq;
  sta->v.n.cthdot = sign(cthdot_sign)*sqrt((geo->carter - geo->masssq*blk->a2*square(cth))*(1-square(cth)) - square(blk->a*geo->energy*(1-square(cth)) - geo->angmom))/rhosq;
#if DO_USELESS
  sta->v.n.l = sqrt((1-square(cth))*(square(r)+blk->a2));
#endif
  sta->coords = coords;
  sta->v.n.t = t;
  sta->v.n.phi = phi;
#if DO_USELESS
  sta->v.n.cphsth = cos(phi)*sqrt(1-square(cth));
  sta->v.n.sphsth = sin(phi)*sqrt(1-square(cth));
#endif
  sta->nbsteps = 0;
}

void
reinit_state (struct state_s *sta)
// Reinitialize state structure from the first integrals of motion.
{
  if ( fabs(sta->v.n.rdot) < 1.e-3 || fabs(sta->v.n.cthdot) < 1.e-3 )
    return; // Avoid region of degeneracy of first-order equation.
  int nbsteps = sta->nbsteps;
  init_state (sta->geo, sta->time, sta->v.n.r, sta->v.n.cth,
	      (sta->v.n.rdot<0), (sta->v.n.cthdot<0), sta->coords,
	      sta->v.n.t,
	      sta->v.n.phi, sta);
  sta->nbsteps = nbsteps;
}

void
equation (const struct state_s *sta, const union variables_u *v,
	  union variables_u *vdot)
// Return the equation to be integrated in vdot (derivatives of all
// variables).  Note: we use the variables in v, not those in sta->v.
{
  const struct blackhole_s *blk = sta->blk;
  const struct geodesic_s *geo = sta->geo;
  double r = v->n.r;
  double cth = v->n.cth;
  double rdot = v->n.rdot;
  double cthdot = v->n.cthdot;
  double rhosq = square(r) + blk->a2*square(cth);
  double rhosqdot = 2*(r*rdot + blk->a2*cth*cthdot);
  vdot->n.r = rdot;
  vdot->n.cth = cthdot;
  double tmpr = geo->carter*blk->M + geo->expr1*r
    + 3*blk->M*geo->masssq*square(r) + geo->expr2*square(r)*r;
  vdot->n.rdot = ((tmpr/rhosq)-rhosqdot*rdot)/rhosq;
  double tmpw = (geo->expr1 - blk->a2*geo->expr2*square(cth))*cth;
  vdot->n.cthdot = ((tmpw/rhosq)-rhosqdot*cthdot)/rhosq;
#if DO_USELESS
  vdot->n.l = (r*rdot*(1-square(cth)) - (square(r)+blk->a2)*cth*cthdot)/v->n.l;
#endif
  double delta = square(r) - 2*blk->M*r + blk->a2;
  double p = (square(r)+blk->a2)*geo->energy - blk->a*geo->angmom;
  assert (sta->coords == 0 || sta->coords == 1);
  if ( fabs(delta)<1.e-6 && ( sta->coords == ((rdot > 0) ^ (p < 0)) ) )
    {
      // We are near the horizon, so delta is very small: use an
      // expansion in delta to avoid numeric loss of precision.
      double c = (geo->masssq*square(r) + geo->carter) / square(p);
      vdot->n.t = (blk->a*geo->angmom - blk->a2*geo->energy*(1-square(cth)))/rhosq
	+ (square(r)+blk->a2)*(c/2 + c*c*delta/8 + c*c*c*delta/16)
	- sign(sta->coords)*rdot;
      vdot->n.phi = (geo->angmom/(1-square(cth)) - blk->a*geo->energy)/rhosq
	+ blk->a*(p/rhosq)*(c/2 + c*c*delta/8 + c*c*c*delta/16);
    }
  else
    {
      // (Normal path of execution)
      vdot->n.t = (blk->a*geo->angmom - blk->a2*geo->energy*(1-square(cth)))/rhosq
	+ (square(r)+blk->a2)*(p/rhosq+sign(sta->coords)*rdot)/delta
	- sign(sta->coords)*rdot;
      vdot->n.phi = (geo->angmom/(1-square(cth)) - blk->a*geo->energy)/rhosq
	+ blk->a*(p/rhosq+sign(sta->coords)*rdot)/delta;
    }
#if DO_USELESS
  vdot->n.cphsth = -v->n.sphsth * vdot->n.phi - v->n.cphsth*(cth/(1-square(cth)))*cthdot;
  vdot->n.sphsth = v->n.cphsth * vdot->n.phi - v->n.sphsth*(cth/(1-square(cth)))*cthdot;
#endif
}

void
runge_kutta (const struct state_s *sta0, double step, union variables_u *vnew)
// Perform a fourth-order Runge-Kutta integration.
{
  const union variables_u *v0 = &(sta0->v);
  union variables_u vdot1;
  union variables_u v1;
  union variables_u vdot2;
  union variables_u v2;
  union variables_u vdot3;
  union variables_u v3;
  union variables_u vdot4;
  equation (sta0, v0, &vdot1);
  for ( int i=0 ; i<NBVARS ; i++ )
    v1.t[i] = v0->t[i] + vdot1.t[i]*step/2;
  equation (sta0, &v1, &vdot2);
  for ( int i=0 ; i<NBVARS ; i++ )
    v2.t[i] = v0->t[i] + vdot2.t[i]*step/2;
  equation (sta0, &v2, &vdot3);
  for ( int i=0 ; i<NBVARS ; i++ )
    v3.t[i] = v0->t[i] + vdot3.t[i]*step;
  equation (sta0, &v3, &vdot4);
  for ( int i=0 ; i<NBVARS ; i++ )
    vnew->t[i] = v0->t[i] + (vdot1.t[i]+vdot2.t[i]*2+vdot3.t[i]*2+vdot4.t[i])/6*step;
}

void
evolve (double step, struct state_s *sta)
// Evolve the state.
{
  union variables_u vnew;
  change_coords_to_best (sta);
  runge_kutta (sta, step, &vnew);
  sta->time += step;
  sta->v = vnew;
  sta->nbsteps++;
}

void
compute_metric (const struct blackhole_s *blk, char coords,
		double r, double cth, double g[4][4])
// Coordinates are in this oder: [0]:r, [1]:cth=cos(theta), [2]:t, [3]:phi.
{
  double rhosq = square(r) + blk->a2*square(cth);
  double sth2 = 1-square(cth);
  double poten = 2*blk->M*r/rhosq;
  g[2][2] = -(1-poten);
  g[1][1] = rhosq/sth2;
  g[3][3] = (square(r) + blk->a2 + poten*blk->a2*sth2)*sth2;
  if ( coords == 0 || coords == 1 )
    {
      g[0][0] = (1+poten);
      g[0][2] = g[2][0] = sign(coords)*poten;
      g[0][3] = g[3][0] = -sign(coords)*(1+poten)*blk->a*sth2;
    }
  else
    {
      g[0][0] = rhosq/(square(r) - 2*blk->M*r + blk->a2);
      g[0][2] = g[2][0] = 0;
      g[0][3] = g[3][0] = 0;
    }
  g[2][3] = g[3][2] = -poten*blk->a*sth2;
  g[0][1] = g[1][0] = 0;
  g[1][2] = g[2][1] = 0;
  g[1][3] = g[3][1] = 0;
}

double
dotprod (double g[4][4], double u[4], double v[4])
// Compute the dot product of two vectors, using the metric g.
{
  double dot = 0.;
  for ( int i=0 ; i<4 ; i++ )
    for ( int j=0 ; j<4 ; j++ )
      if ( u[i] != 0. && v[j] != 0. ) // Avoid irritating NaNs.
	dot += g[i][j] * u[i] * v[j];
  return dot;
}

void
init_from_v_and_vdot (const struct blackhole_s *blk, char coords,
		      double r, double cth, double t, double phi,
		      double rdot, double cthdot, double tdot, double phidot,
		      char normalize_masssq,
		      struct geodesic_s *geo, struct state_s *sta)
// Initialize the geodisc (conserved quantities) and the current
// state, simultaneously, from given position and velocity.  coords
// indicates whether we are using ingoing (0), outgoing (1) or
// Boyer-Lindquist (2) coordinates, and normalize_masssq should be set
// to -1, 0 or 1 if the geodesic is known to be spacelike (-1),
// lightlike (0) or timelike (1), or to 2 to avoid making an
// assumption or normalizing the given velocity.
{
  double g[4][4];
  compute_metric (blk, coords, r, cth, g);
  double vec[4] = { rdot, cthdot, tdot, phidot };
  double dbydt[4] = { 0, 0, 1, 0 };
  double dbydphi[4] = { 0, 0, 0, 1 };
  double masssq = -dotprod(g, vec, vec);
  if ( normalize_masssq == -1 || normalize_masssq == 1 ) {
    // Normalize velocity.
    double t = sqrt(masssq/normalize_masssq);
    rdot /= t;  cthdot /= t;  tdot /= t;  phidot /= t;
    for ( int i=0 ; i<4 ; i++ )
      vec[i] /= t;
    masssq = normalize_masssq;
  } else if ( normalize_masssq == 0 ) {
    // Assume lightlike.
    masssq = 0;  // Not a great idea?
  }
  double energy = -dotprod(g, vec, dbydt);
  double angmom = dotprod(g, vec, dbydphi);
  double rhosq = square(r) + blk->a2*square(cth);
  double carter = (square(rhosq)*square(cthdot) + square(blk->a*energy*(1-square(cth))-angmom))/(1-square(cth)) + masssq*blk->a2*square(cth);
  init_geodesic (blk, masssq, energy, angmom, carter, geo);
  init_state (geo, 0, r, cth, rdot<0, cthdot<0, coords, t, phi, sta);
  sta->v.n.rdot = rdot;
  sta->v.n.cthdot = cthdot;
}

char
checker (double cth, double phi)
// Controls how the spheres are checkered (return 1 when a band is met
// for the given cos(theta) and phi).  Note that rotation of horizons
// is handled by the caller!
{
  double turns = phi/(2*M_PI);
  turns *= 24;
  turns -= floor(turns);
  if ( turns<0.1 || turns>0.9 )
    return 1;
  double bands = acos(cth)/(M_PI);
  bands *= 12;
  bands -= floor(bands);
  if ( bands<0.1 || bands>0.9 )
    return 1;
  return 0;
}

char
check_crossing (struct state_s *oldstate, struct state_s *newstate,
		double rtgt, char force_coords,
		struct state_s *xstate, char *pdir)
// Check whether we have crossed the sphere with r=rtgt between
// oldstate and newstate and, if so, accurately compute the point of
// crossing.  The crossing state is returned in xstate, and the
// direction of crossing in *pdir.
{
  char dir = 0;
  if ( oldstate->v.n.r < rtgt && newstate->v.n.r >= rtgt )
    dir = 1;
  else if ( oldstate->v.n.r > rtgt && newstate->v.n.r <= rtgt )
    dir = -1;
  else
    {
      if ( pdir )
	*pdir = 0;
      return 0;
    }
  struct state_s state;
  memcpy (&state, oldstate, sizeof(struct state_s));
  for ( int i=0 ; i<4 ; i++ )
    {
      double xstep = (rtgt - state.v.n.r) / state.v.n.rdot;
      if ( fabs(xstep) < 1.e-8 )
	break;
      evolve (xstep, &state);
    }
  if ( force_coords >= 0 )
    change_coords (&state, force_coords);
  memcpy (xstate, &state, sizeof(struct state_s));
  if ( pdir )
    *pdir = dir;
  return dir;
}

struct color_s { int red; int green; int blue; };

const struct color_s lost_color = { 0, 0, 0 };
const struct color_s nan_color = { 64, 64, 64 };

const struct color_s outer_sphere_color = { 0, 0, 128 };
const struct color_s outer_sphere_checker = { 0, 0, 255 };
const struct color_s foreign_sphere_color = { 0, 64, 128 };
const struct color_s foreign_sphere_checker = { 0, 128, 255 };
const struct color_s negative_sphere_color = { 128, 0, 128 };
const struct color_s negative_sphere_checker = { 255, 0, 255 };

const struct color_s outer_horizon_outgoing_outcrossing_checker = { 255, 0, 0 };
const struct color_s outer_horizon_ingoing_outcrossing_checker = { 255, 128, 0 };
const struct color_s outer_horizon_outgoing_incrossing_checker = { 128, 0, 0 };
const struct color_s outer_horizon_ingoing_incrossing_checker = { 128, 64, 0 };
const struct color_s inner_horizon_outgoing_outcrossing_checker = { 0, 255, 0 };
const struct color_s inner_horizon_ingoing_outcrossing_checker = { 0, 255, 128 };
const struct color_s inner_horizon_outgoing_incrossing_checker = { 0, 128, 0 };
const struct color_s inner_horizon_ingoing_incrossing_checker = { 0, 128, 64 };

int
main (int argc, char *argv[])
{
  struct blackhole_s blk;
  init_blackhole (BLACKHOLE_MASS, BLACKHOLE_ANGMOM_PER_MASS, &blk);
  double g[4][4];
  double time0; // (ignored)
  int coords0;
  double r0;
  double cth0;
  double t0;
  double phi0;
  scanf (" %lf", &time0); // (ignored)
  scanf (" %d", &coords0);
  scanf (" %lf %lf %lf %lf", &r0, &cth0, &t0, &phi0);
  compute_metric (&blk, coords0, r0, cth0, g);
#if 0
  double vecs[4][4] = {
    { 0, 0, 1, 0 },  // Time-pointing vector of observer
    { -1, 0, 0, 0 }, // Space forward-pointing vector of observer
    { 0, 0, 0, 1 },  // Space right-pointing vector of observer
    { 0, 1, 0, 0 }   // Space up-pointing vector of observer
  };
#else
  double vecs[4][4];
  for ( int m=0 ; m<4 ; m++ )
    scanf (" %lf %lf %lf %lf",
	   &vecs[m][0], &vecs[m][1], &vecs[m][2], &vecs[m][3]);
#endif
  fprintf (stderr, "Position:\n%.8e\t%.8e\t%.8e\t%.8e\n",
	   r0, cth0, t0, phi0);
  fprintf (stderr, "Before orthonormalization:\n");
  for ( int m=0 ; m<4 ; m++ )
    fprintf (stderr, "%.8e\t%.8e\t%.8e\t%.8e\n",
	     vecs[m][0], vecs[m][1], vecs[m][2], vecs[m][3]);
  for ( int m=0 ; m<4 ; m++ )
    {  // Gram-Schmidt orthonormalization
      for ( int mm=0 ; mm<m ; mm++ )
	{
	  double s = dotprod(g, vecs[m], vecs[mm]);
	  for ( int i=0 ; i<4 ; i++ )
	    vecs[m][i] -= sign(mm==0)*s*vecs[mm][i];
	}
      double norm = sqrt(sign(m==0)*dotprod(g, vecs[m], vecs[m]));
      for ( int i=0 ; i<4 ; i++ )
	vecs[m][i] /= norm;
    }
  fprintf (stderr, "After orthonormalization:\n");
  for ( int m=0 ; m<4 ; m++ )
    fprintf (stderr, "%.8e\t%.8e\t%.8e\t%.8e\n",
	     vecs[m][0], vecs[m][1], vecs[m][2], vecs[m][3]);
  int lin_min = argc>=2?atoi(argv[1]):0;
  int lin_max = argc>=3?atoi(argv[2]):NBLINS;
  fprintf (stderr, "Computing image from line %d to line %d...\n",
	   lin_min, lin_max);
  if ( lin_min == 0 )
    printf ("P3\n%d %d\n255\n", NBCOLS, NBLINS);
  for ( int lin=lin_min ; lin<lin_max ; lin++ )
    {
      fprintf (stderr, "%d/%d lines done of %d total\n", (lin-lin_min), (lin_max-lin_min), NBLINS);
      for ( int col=0 ; col<NBCOLS ; col++ )
	{
	  double x = (col-(NBCOLS-1)/2.)/(NBLINS/2.4);
	  double y = ((NBLINS-1)/2.-lin)/(NBLINS/2.4);
	  x /= ZOOM_FACTOR;  y /= ZOOM_FACTOR;
	  double tmp = sqrt(1+square(x)+square(y));
	  x /= tmp;
	  y /= tmp;
	  double z = 1/tmp;
	  struct geodesic_s geo;
	  struct state_s sta, oldsta, xsta;
	  // Throw a light ray (toward the past).
	  init_from_v_and_vdot (&blk, coords0, r0, cth0, t0, phi0,
				-vecs[0][0]+z*vecs[1][0]+x*vecs[2][0]+y*vecs[3][0],
				-vecs[0][1]+z*vecs[1][1]+x*vecs[2][1]+y*vecs[3][1],
				-vecs[0][2]+z*vecs[1][2]+x*vecs[2][2]+y*vecs[3][2],
				-vecs[0][3]+z*vecs[1][3]+x*vecs[2][3]+y*vecs[3][3],
				0, // <- Light rays are lightlike.
				&geo, &sta);
	  struct color_s color;
	  int horizon_crossings = 0;
	  while ( 1 )
	    {
	      if ( isnan(sta.v.n.r) || isnan(sta.v.n.cth) )
		{ color = nan_color;  break; }
	      if ( sta.nbsteps > MAX_NBSTEPS || sta.time > MAX_PROPERTIME
		   || horizon_crossings >= 4 )
		{ color = lost_color;  break; }
#if 1
	      // Periodically recontrol rdot and cthdot.
	      if ( sta.nbsteps % 512 == 511 )
		reinit_state (&sta);
#endif
	      memcpy (&oldsta, &sta, sizeof(struct state_s));
	      // A heuristic on step size:
	      double step = BASIC_STEP * ((1+fabs(oldsta.v.n.r))
					  / (1+fabs(oldsta.v.n.rdot)));
	      if ( oldsta.v.n.cth > 0.95 || oldsta.v.n.cth < -0.95 )
		step /= 3; // Increase accuracy near the axis.
	      evolve (step, &sta);
	      char dir;
	      if ( check_crossing (&oldsta, &sta, FARSPHERE_RADIUS, 2,
				   &xsta, NULL) )
		{
		  // Compute tdot, so we can color spheres slightly
		  // differently depending on its value:
		  double r = xsta.v.n.r;
		  double cth = xsta.v.n.cth;
		  double rdot = xsta.v.n.rdot;
		  double rhosq = square(r) + blk.a2*square(cth);
		  double delta = square(r) - 2*blk.M*r + blk.a2;
		  double p = (square(r)+blk.a2)*geo.energy - blk.a*geo.angmom;
		  double tdot = (blk.a*geo.angmom - blk.a2*geo.energy*(1-square(cth)))/rhosq
		    + (square(r)+blk.a2)*(p/rhosq+sign(xsta.coords)*rdot)/delta
		    - sign(xsta.coords)*rdot;
		  if ( checker (xsta.v.n.cth, xsta.v.n.phi) )
		    color = (tdot>0) ? foreign_sphere_checker
		      : outer_sphere_checker;
		  else
		    color = (tdot>0) ? foreign_sphere_color
		      : outer_sphere_color;
		  memcpy (&sta, &xsta, sizeof(struct state_s));
		  break;
		}
	      if ( check_crossing (&oldsta, &sta, blk.M+blk.sqrtM2ma2, -1, 
				   &xsta, &dir) )
		{
		  // Crossed outer horizon.
		  if ( checker (xsta.v.n.cth, xsta.v.n.phi
				- xsta.v.n.t * (blk.a/(2*blk.M*xsta.v.n.r))
				) )
		    {
		      if ( dir < 0 && xsta.coords )
			color = outer_horizon_outgoing_outcrossing_checker;
		      else if ( dir > 0 && xsta.coords )
			color = outer_horizon_outgoing_incrossing_checker;
		      else if ( dir < 0 )
			color = outer_horizon_ingoing_outcrossing_checker;
		      else if ( dir > 0 )
			color = outer_horizon_ingoing_incrossing_checker;
		      else
			assert ( 0 );
		      memcpy (&sta, &xsta, sizeof(struct state_s));
		      break;
		    }
		  horizon_crossings++;
		}
	      if ( check_crossing (&oldsta, &sta, blk.M-blk.sqrtM2ma2, -1, 
				   &xsta, &dir) )
		{
		  // Crossed inner horizon.
		  if ( checker (xsta.v.n.cth, xsta.v.n.phi
				- xsta.v.n.t * (blk.a/(2*blk.M*xsta.v.n.r))
				) )
		    {
		      if ( dir < 0 && xsta.coords )
			color = inner_horizon_outgoing_outcrossing_checker;
		      else if ( dir > 0 && xsta.coords )
			color = inner_horizon_outgoing_incrossing_checker;
		      else if ( dir < 0 )
			color = inner_horizon_ingoing_outcrossing_checker;
		      else if ( dir > 0 )
			color = inner_horizon_ingoing_incrossing_checker;
		      else
			assert ( 0 );
		      memcpy (&sta, &xsta, sizeof(struct state_s));
		      break;
		    }
		  horizon_crossings++;
		}
	      if ( check_crossing (&oldsta, &sta, NEGSPHERE_RADIUS, 2,
				   &xsta, NULL) )
		{
		  if ( checker (xsta.v.n.cth, xsta.v.n.phi) )
		    color = negative_sphere_checker;
		  else
		    color = negative_sphere_color;
		  memcpy (&sta, &xsta, sizeof(struct state_s));
		  break;
		}
	    }
	  printf ("  %3d %3d %3d", color.red, color.green, color.blue);
#if 1
	  if ( rand()%1000 == 0 )
	    {
	      fprintf (stderr, "Random point (%d,%d) dump: color is (%d,%d,%d)\n",
		       col, lin, color.red, color.green, color.blue);
	      fprintf (stderr, "Number of steps: %d\n", sta.nbsteps);
	      fprintf (stderr, "Final proper time: %.7f\n", sta.time);
	      fprintf (stderr, "Final r: %.7f\n", sta.v.n.r);
	      fprintf (stderr, "***\n");
	    }
#endif
	}
      printf ("\n");
    }
  return 0;
}
