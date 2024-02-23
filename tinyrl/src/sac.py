import os, sys
from torch.utils.cpp_extension import load
from pkg_resources import resource_filename

import pkg_resources

extra_ldflags = []
extra_include_paths = []
extra_cflags = []
force_mkl = "TINYRL_FORCE_MKL" in os.environ
if sys.platform == "linux":
    try:
        mkl_version = pkg_resources.get_distribution("mkl").version
        mkl_include_version = pkg_resources.get_distribution("mkl-include").version
        print(f"MKL is installed. Version: {mkl_version} (include: {mkl_include_version})")
        mkl_lib_path = os.path.join(sys.prefix, "lib")
        # create version symlinks for the MKL libraries
        for mkl_lib in ["libmkl_intel_ilp64.so", "libmkl_intel_thread.so", "libmkl_core.so"]:
            source = os.path.join(mkl_lib_path, mkl_lib + ".2")
            target = os.path.join(mkl_lib_path, mkl_lib)
            print("checking: " + source)
            assert(os.path.exists(source))
            if not os.path.exists(target):
                os.symlink(source, target)
        extra_ldflags += [
            "-m64",
            "-Wl,--no-as-needed",
            "-lmkl_intel_ilp64",
            "-lmkl_intel_thread",
            "-lmkl_core",
            "-liomp5",
            "-lpthread",
            "-lm",
            "-ldl",
            "-L" + mkl_lib_path,
            "-Wl,--rpath," + mkl_lib_path,
        ]
        extra_include_paths += [os.path.join(sys.prefix + "/include")]
        extra_cflags += ["-DRL_TOOLS_BACKEND_ENABLE_MKL"]
    except pkg_resources.DistributionNotFound:
        assert(not force_mkl)
        print("MKL is not installed.")


enable_optimization = True

absolute_path = os.path.dirname(os.path.abspath(__file__))

def loop_sac(env_factory):

    cpp_std_flag = '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17'
    optimization_flag = ('-O3' if not sys.platform.startswith('win') else '/O2') if enable_optimization else ''
    arch_flags = '-march=native' if not sys.platform.startswith('win') else '/arch:AVX2'
    fast_math_flag = '-ffast-math' if not sys.platform.startswith('win') else '/fp:fast'

    observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={env_factory().observation_space.shape[0]}'
    action_dim_flag = f'-DTINYRL_ACTION_DIM={env_factory().action_space.shape[0]}'
    print(f"Compiling the TinyRL interface...")
    loop = load(
        'rl_tools',
        sources=[os.path.join(absolute_path, '../interface/python_environment/python_environment.cpp')],
        extra_include_paths=[
            os.path.join(absolute_path, "..", "external", "rl_tools", "include"),
            *extra_include_paths
        ],
        extra_cflags=[cpp_std_flag, optimization_flag, arch_flags, fast_math_flag, observation_dim_flag, action_dim_flag, *extra_cflags],
        extra_ldflags=[*extra_ldflags]
    )
    print(f"Finished compiling the TinyRL interface.")


    loop.set_environment_factory(env_factory)
    return loop