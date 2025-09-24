#ifndef PPO_AGENT_CONTINUOUS_H
#define PPO_AGENT_CONTINUOUS_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <algorithm>
#include <memory>

// Namespace for internal logic of the continuous version
namespace PPOInternalContinuous {
    class NeuralNetworkContinuous;
    class AdamOptimizerContinuous;
    class ReplayBufferContinuous;
    struct TransitionContinuous;
    struct PPOCoreConfigContinuous;
    class PPOCoreContinuous;
}

class PPOC : public godot::RefCounted {
    GDCLASS(PPOC, godot::RefCounted);

private:
    std::unique_ptr<PPOInternalContinuous::PPOCoreContinuous> ppo_core_continuous_;
    std::unique_ptr<PPOInternalContinuous::ReplayBufferContinuous> replay_buffer_continuous_;
    
    Eigen::VectorXf current_observation_continuous_;
    Eigen::VectorXf current_action_vector_continuous_;

    float current_action_log_prob_continuous_;
    float current_value_estimate_continuous_;

    bool initialized_continuous_ = false;
    int observation_dim_continuous_;
    int action_dim_continuous_;
    int buffer_size_continuous_;
    int training_counter_continuous_ = 0;
    int train_every_n_steps_continuous_ = 2048;

protected:
    static void _bind_methods();

public:
    PPOC();
    ~PPOC();

    void initialize(const godot::Dictionary& config);
    godot::PackedFloat32Array get_action(const godot::PackedFloat32Array& observation_array);
    void store_experience(float reward, const godot::PackedFloat32Array& next_observation_array, bool done);
    void train();

    bool save_model(const godot::String& file_path);
    bool load_model(const godot::String& file_path);

    // Helper functions for type conversions
    static Eigen::VectorXf packed_array_to_eigen_vector(const godot::PackedFloat32Array& p_array);
    static godot::PackedFloat32Array eigen_vector_to_packed_array(const Eigen::VectorXf& p_vector);
    static godot::PackedByteArray eigen_vector_to_byte_array(const Eigen::VectorXf& vec);
    static Eigen::VectorXf byte_array_to_eigen_vector(const godot::PackedByteArray& byte_arr, int size);
    static godot::PackedByteArray eigen_matrix_to_byte_array(const Eigen::MatrixXf& mat);
    static Eigen::MatrixXf byte_array_to_eigen_matrix(const godot::PackedByteArray& byte_arr, int rows, int cols);
};

#endif // PPO_AGENT_CONTINUOUS_H