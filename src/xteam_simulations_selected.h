#pragma once
#include "xteam_simulations_common.h"

#ifdef AOMP
#include "xteam_simulations_aomp.h"
template <typename T> using SelectedSim = SimulationAOMP<T>;
#elif defined(TRUNK)
#include "xteam_simulations_trunk.h"
template <typename T> using SelectedSim = SimulationTrunk<T>;
#elif defined(TRUNK_JD)
#include "xteam_simulations_trunk_jd.h"
template <typename T> using SelectedSim = SimulationTrunkJD<T>;
#elif defined(TRUNK_DEV)
#include "xteam_simulations_trunk_dev.h"
template <typename T> using SelectedSim = SimulationTrunkDev<T>;
#elif defined(AOMP_DEV)
#include "xteam_simulations_aomp_dev.h"
template <typename T> using SelectedSim = SimulationAOMPDev<T>;
#else
template <typename T> using SelectedSim = SimulationNoop<T>;
#endif
