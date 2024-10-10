#ifndef RL_TOOLS_MODULE_NAME
#error "RL_TOOLS_MODULE_NAME not defined"
#endif

#ifndef RL_TOOLS_DTYPE
#define RL_TOOLS_DTYPE float
#endif

#include <rl_tools/operations/cpu_mux.h>

#ifdef RL_TOOLS_USE_PYTHON_ENVIRONMENT
#include "../python_environment/operations_cpu.h"
#else
#include <environment.h>
#endif

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include "loop_core_config.h"

namespace rlt = rl_tools;

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace RL_TOOLS_MODULE_NAME{
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#ifdef RL_TOOLS_FORCE_BLAS
    static_assert(DEVICE::DEVICE_ID == rlt::devices::DeviceId::CPU_MKL || DEVICE::DEVICE_ID == rlt::devices::DeviceId::CPU_ACCELERATE || DEVICE::DEVICE_ID == rlt::devices::DeviceId::CPU_OPENBLAS);
#endif
    using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
    using TI = typename DEVICE::index_t;

    using T = RL_TOOLS_DTYPE;


    #ifdef RL_TOOLS_USE_PYTHON_ENVIRONMENT
    constexpr TI OBSERVATION_DIM = RL_TOOLS_OBSERVATION_DIM;
    constexpr TI ACTION_DIM = RL_TOOLS_ACTION_DIM;
    #ifdef RL_TOOLS_EPISODE_STEP_LIMIT
    constexpr TI EPISODE_STEP_LIMIT = RL_TOOLS_EPISODE_STEP_LIMIT;
    #else
    constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
    #endif
    using ENVIRONMENT_SPEC = PythonEnvironmentSpecification<T, TI, OBSERVATION_DIM, ACTION_DIM, EPISODE_STEP_LIMIT>;
    using ENVIRONMENT = PythonEnvironment<ENVIRONMENT_SPEC>;
    #else
    using ENVIRONMENT = ENVIRONMENT_FACTORY<T, TI>;
    #endif




    using LOOP_CORE_CONFIG = LOOP_CORE_CONFIG_FACTORY<T, TI, RNG, ENVIRONMENT>;



    DEVICE device;

    #ifdef RL_TOOLS_ENABLE_EVALUATION
    constexpr bool ENABLE_EVALUATION = true;
    #ifndef RL_TOOLS_EVALUATION_INTERVAL
    #error "RL_TOOLS_EVALUATION_INTERVAL not defined"
    #else
    constexpr TI PARAMETER_EVALUATION_INTERVAL = RL_TOOLS_EVALUATION_INTERVAL;
    constexpr TI PARAMETER_NUM_EVALUATION_EPISODES = RL_TOOLS_NUM_EVALUATION_EPISODES;

    template <typename NEXT>
    struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
        static constexpr TI EVALUATION_INTERVAL = PARAMETER_EVALUATION_INTERVAL;
        static constexpr TI NUM_EVALUATION_EPISODES = PARAMETER_NUM_EVALUATION_EPISODES;
        static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
    };
    using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
    #endif
    #else
    constexpr bool ENABLE_EVALUATION = false;
    using LOOP_EVAL_CONFIG = LOOP_CORE_CONFIG;
    #endif
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
    using LOOP_CONFIG = LOOP_TIMING_CONFIG;
    using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;


    constexpr TI EVALUATION_BATCH_SIZE = 1;
    using ACTOR_TYPE = rlt::utils::typing::remove_reference_t<decltype(rlt::get_actor(std::declval<LOOP_STATE&>()))>;
    using EVALUATION_ACTOR_TEMP = ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, EVALUATION_BATCH_SIZE>;
    using EVALUATION_ACTOR = EVALUATION_ACTOR_TEMP::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
    using EVALUATION_ACTOR_STATE = EVALUATION_ACTOR::State<>;
    using EVALUATION_ACTOR_BUFFER = EVALUATION_ACTOR::Buffer<>;

    #ifdef RL_TOOLS_USE_PYTHON_ENVIRONMENT
    void set_environment_factory(std::function<pybind11::object()> p_environment_factory){
        environment_factory = p_environment_factory;
        auto python_atexit = pybind11::module_::import("atexit");
        python_atexit.attr("register")(pybind11::cpp_function([]() {
            environment_factory = nullptr;
        }));
    }
    #endif
    struct State: LOOP_STATE{
        EVALUATION_ACTOR evaluation_actor;
        EVALUATION_ACTOR_STATE evaluation_actor_state;
        EVALUATION_ACTOR_BUFFER evaluation_actor_buffer;
        bool evaluation_actor_synced = false;
        State(TI seed){
            rlt::malloc(device, static_cast<LOOP_STATE&>(*this));
            rlt::init(device, static_cast<LOOP_STATE&>(*this), seed);
            rlt::malloc(device, evaluation_actor);
            rlt::malloc(device, evaluation_actor_state);
            rlt::malloc(device, evaluation_actor_buffer);
            evaluation_actor_synced = false;
        }
        bool step(){
            evaluation_actor_synced = false;
            return rlt::step(device, static_cast<LOOP_STATE&>(*this));
        }
        pybind11::array_t<T> action(const pybind11::array_t<T>& observation){
            if(!evaluation_actor_synced){
                rlt::copy(device, device, rlt::get_actor(*this), evaluation_actor);
                evaluation_actor_synced = true;
            }
            pybind11::buffer_info observation_info = observation.request();
            if (observation_info.format != pybind11::format_descriptor<T>::format() || observation_info.ndim != 1) {
                throw std::runtime_error("Incompatible buffer format. Check the floating point type of the observation returned by env.step() and the one configured when building the RLtools interface");
            }
            auto observation_data_ptr = static_cast<T*>(observation_info.ptr);
            size_t num_elements = observation_info.shape[0];
            if(num_elements != ENVIRONMENT::Observation::DIM){
                throw std::runtime_error("Incompatible observation dimension. Check the dimension of the observation returned by env.step() and the one configured when building the RLtools interface");
            }
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, EVALUATION_BATCH_SIZE, ENVIRONMENT::Observation::DIM>, false>> observation_rlt;
            for(TI observation_i=0; observation_i<num_elements; observation_i++){
                rlt::set(device, observation_rlt, observation_data_ptr[observation_i], 0, observation_i);
            }
            // using ACTOR_TYPE = rlt::utils::typing::remove_reference<decltype(rlt::get_actor(*this))>::type;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, EVALUATION_BATCH_SIZE, ENVIRONMENT::ACTION_DIM>, false>> action_rlt; //2x for mean and std
            bool rng = false;
            rlt::Mode<rlt::mode::Evaluation<>> mode;
            rlt::evaluate_step(device, evaluation_actor, observation_rlt, evaluation_actor_state, action_rlt, evaluation_actor_buffer, rng, mode);

            std::vector<T> action(ENVIRONMENT::ACTION_DIM);

            for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                action[action_i] = rlt::get(device, action_rlt, 0, action_i);
            }

            return pybind11::array_t<T>(ENVIRONMENT::ACTION_DIM, action.data());
        }
        std::string export_policy(){
            rlt::copy(device, device, rlt::get_actor(*this), evaluation_actor);
            return rlt::save_code(device, evaluation_actor, "policy");
        }
        ~State(){
            rlt::free(device, static_cast<LOOP_STATE&>(*this));
            rlt::malloc(device, evaluation_actor);
            rlt::malloc(device, evaluation_actor_state);
            rlt::malloc(device, evaluation_actor_buffer);
        }
    };
}




PYBIND11_MODULE(RL_TOOLS_MODULE_NAME, m){
    m.doc() = "RLtools Training Loop";
    pybind11::class_<RL_TOOLS_MODULE_NAME::State>(m, "State")
            .def(pybind11::init<RL_TOOLS_MODULE_NAME::TI>())
            .def("step", &RL_TOOLS_MODULE_NAME::State::step, "Step the loop")
            .def("action", &RL_TOOLS_MODULE_NAME::State::action, "Get the action for the given observation")
            .def("export_policy", &RL_TOOLS_MODULE_NAME::State::export_policy, "Export the policy to a python file");
#ifdef RL_TOOLS_USE_PYTHON_ENVIRONMENT
    m.def("set_environment_factory", &RL_TOOLS_MODULE_NAME::set_environment_factory, "Set the environment factory");
#endif
}