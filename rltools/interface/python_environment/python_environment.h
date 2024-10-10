#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <rl_tools/rl/environments/environments.h>


template <typename T_T, typename T_TI, T_TI T_OBSERVATION_DIM, T_TI T_ACTION_DIM, T_TI T_EPISODE_STEP_LIMIT>
struct PythonEnvironmentSpecification{
    using T = T_T;
    using TI = T_TI;
    static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
    static constexpr TI ACTION_DIM = T_ACTION_DIM;
    static constexpr TI EPISODE_STEP_LIMIT = T_EPISODE_STEP_LIMIT;
};

template <typename T_T, typename T_TI>
struct PythonEnvironmentParameters{
};

template <typename T_T, typename T_TI, T_TI T_DIM>
struct PythonEnvironmentState{
    using T = T_T;
    using TI = T_TI;
    static constexpr TI DIM = T_DIM;
    std::array<T, DIM> state;
    T reward;
    bool terminated;
};

template <typename T_SPEC>
struct PythonEnvironmentObservation{
    static constexpr typename T_SPEC::TI DIM = T_SPEC::OBSERVATION_DIM;
};

template <typename T_SPEC>
struct PythonEnvironment: rl_tools::rl::environments::Environment<typename T_SPEC::T, typename T_SPEC::TI>{
    using SPEC = T_SPEC;
    using T = typename SPEC::T;
    using TI = typename SPEC::TI;
    using Parameters = PythonEnvironmentParameters<T, TI>;
    using State = PythonEnvironmentState<T, TI, SPEC::OBSERVATION_DIM>;
    using Observation = PythonEnvironmentObservation<SPEC>;
    using ObservationPrivileged = Observation;
    static constexpr TI ACTION_DIM = SPEC::ACTION_DIM;
    pybind11::object* environment;
    static constexpr TI EPISODE_STEP_LIMIT = SPEC::EPISODE_STEP_LIMIT;
    // T action_space_range[ACTION_DIM];
    // T action_space_offset[ACTION_DIM];
};