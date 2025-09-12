#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace RL_TOOLS_MODULE_NAME{

    namespace rlt = rl_tools;

    // This should stay in sync with the parameters in https://github.com/rl-tools/rl-tools/blob/a98bc461f7ebfed0ba71c653216edec6d9334b78/include/rl_tools/rl/algorithms/td3/loop/core/config.h#L18
    template <typename T, typename TI, typename ENVIRONMENT>
    struct TD3_LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
        struct TD3_PARAMETERS{
            static constexpr T GAMMA = $GAMMA;
            static constexpr TI ACTOR_BATCH_SIZE = $ACTOR_BATCH_SIZE;
            static constexpr TI CRITIC_BATCH_SIZE = $CRITIC_BATCH_SIZE;
            static constexpr TI CRITIC_TRAINING_INTERVAL = $CRITIC_TRAINING_INTERVAL;
            static constexpr TI ACTOR_TRAINING_INTERVAL = $ACTOR_TRAINING_INTERVAL;
            static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = $CRITIC_TARGET_UPDATE_INTERVAL;
            static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = $ACTOR_TARGET_UPDATE_INTERVAL;
            static constexpr T ACTOR_POLYAK = $ACTOR_POLYAK;
            static constexpr T CRITIC_POLYAK = $CRITIC_POLYAK;
            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = $TARGET_NEXT_ACTION_NOISE_STD;
            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = $TARGET_NEXT_ACTION_NOISE_CLIP;
            static constexpr bool IGNORE_TERMINATION = $IGNORE_TERMINATION;
            static constexpr TI SEQUENCE_LENGTH = 1;
            static constexpr bool MASK_NON_TERMINAL = true;
        };
        static constexpr TI N_ENVIRONMENTS = $N_ENVIRONMENTS;
        static constexpr int N_WARMUP_STEPS = $N_WARMUP_STEPS;
        static constexpr TI STEP_LIMIT = $STEP_LIMIT;
        static constexpr TI REPLAY_BUFFER_CAP = $REPLAY_BUFFER_CAP;
        static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;

        static constexpr TI ACTOR_HIDDEN_DIM = $ACTOR_HIDDEN_DIM;
        static constexpr TI ACTOR_NUM_LAYERS = $ACTOR_NUM_LAYERS;
        static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::$ACTOR_ACTIVATION_FUNCTION;
        static constexpr TI CRITIC_HIDDEN_DIM = $CRITIC_HIDDEN_DIM;
        static constexpr TI CRITIC_NUM_LAYERS = $CRITIC_NUM_LAYERS;
        static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::$CRITIC_ACTIVATION_FUNCTION;

        static constexpr bool COLLECT_EPISODE_STATS = $COLLECT_EPISODE_STATS;
        static constexpr TI EPISODE_STATS_BUFFER_SIZE = $EPISODE_STATS_BUFFER_SIZE;
        static constexpr T EXPLORATION_NOISE = $EXPLORATION_NOISE;
        static constexpr bool SHARED_BATCH = $SHARED_BATCH;
        static constexpr bool SAMPLE_ENVIRONMENT_PARAMETERS = true;

        using INITIALIZER = rlt::nn::layers::dense::DefaultInitializer<T, TI>;

        using BATCH_SAMPLING_PARAMETERS = rlt::rl::components::off_policy_runner::SequentialBatchParameters<T, TI, TD3_PARAMETERS::SEQUENCE_LENGTH, rlt::rl::components::off_policy_runner::SequentialBatchParametersDefault>;

        struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
            static constexpr T ALPHA = $OPTIMIZER_ALPHA;
            static constexpr T BETA_1 = $OPTIMIZER_BETA_1;
            static constexpr T BETA_2 = $OPTIMIZER_BETA_2;
            static constexpr T EPSILON = $OPTIMIZER_EPSILON;
            static constexpr T EPSILON_SQRT = $OPTIMIZER_EPSILON_SQRT;
        };
    };

    template <typename T, typename TI, typename RNG, typename ENVIRONMENT>
    using LOOP_CORE_CONFIG_FACTORY = rlt::rl::algorithms::td3::loop::core::Config<T, TI, RNG, ENVIRONMENT, TD3_LOOP_CORE_PARAMETERS<T, TI, ENVIRONMENT>>;
}