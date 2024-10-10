#include <rl_tools/operations/cpu_mux.h>

#include <checkpoint.h>

#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DEVICE_FACTORY<>;

DEVICE device;
auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{});
using MODEL_TYPE = decltype(policy::module);
using T = typename MODEL_TYPE::T;
using TI = typename DEVICE::index_t;
typename MODEL_TYPE::template Buffer<1> buffer;
bool initialized = false;

void init(){
    if(!initialized){
        rlt::malloc(device, buffer);
    }
}

pybind11::array_t<T> evaluate(const pybind11::array_t<T>& input){
    init();
    pybind11::buffer_info input_info = input.request();
    if (input_info.format != pybind11::format_descriptor<T>::format() || input_info.ndim != 1) {
        throw std::runtime_error("Incompatible buffer format. Check the floating point type of the input and the one configured when building the RLtools interface");
    }
    auto input_data_ptr = static_cast<T*>(input_info.ptr);
    size_t num_elements = input_info.shape[0];
    if(num_elements != MODEL_TYPE::INPUT_SHAPE::GET<2>){
        throw std::runtime_error("Incompatible input dimension. Check the dimension of the input and the one configured when building the RLtools interface");
    }
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL_TYPE::INPUT_SHAPE, false>> input_rlt;
    static_assert(MODEL_TYPE::INPUT_SHAPE::LENGTH == 3);
    static_assert(MODEL_TYPE::INPUT_SHAPE::GET<0> == 1);
    static_assert(MODEL_TYPE::INPUT_SHAPE::GET<1> == 1);
    for(TI input_i=0; input_i<num_elements; input_i++){
        rlt::set(device, input_rlt, input_data_ptr[input_i], 0, 0, input_i);
    }
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL_TYPE::OUTPUT_SHAPE, false>> output_rlt;
    static_assert(MODEL_TYPE::OUTPUT_SHAPE::LENGTH == 3);
    static_assert(MODEL_TYPE::OUTPUT_SHAPE::GET<0> == 1);
    static_assert(MODEL_TYPE::OUTPUT_SHAPE::GET<1> == 1);
    rlt::evaluate(device, policy::module, input_rlt, output_rlt, buffer, rng);

    std::vector<T> output(MODEL_TYPE::OUTPUT_SHAPE::GET<2>);

    for (TI output_i = 0; output_i < MODEL_TYPE::OUTPUT_SHAPE::GET<2>; output_i++) {
        output[output_i] = rlt::get(device, output_rlt, 0, 0, output_i);
    }

    return pybind11::array_t<T>(MODEL_TYPE::OUTPUT_SHAPE::GET<2>, output.data());
}

PYBIND11_MODULE(RL_TOOLS_MODULE_NAME, m){
    m.doc() = "RLtools Inference";
    m.def("evaluate", &evaluate, "Evaluate the NN");
}