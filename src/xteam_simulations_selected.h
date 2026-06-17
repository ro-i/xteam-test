#pragma once
#include "xteam_simulations_common.h"

// The Makefile passes the simulation header for the active label as
// -DXTEAM_SIM_HEADER="xteam_simulations_<label>.h". Each such header defines
// the SelectedSim<T> alias. Without it (e.g. unconfigured build), fall back to
// the no-op simulation.
#ifdef XTEAM_SIM_HEADER
#include XTEAM_SIM_HEADER
#else
template <typename T> using SelectedSim = SimulationNoop<T>;
#endif
