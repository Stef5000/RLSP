#ifndef PPO_DISCRETE_H
#define PPO_DISCRETE_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
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

namespace godot {
    class FileAccess;
}

namespace PPOInternal {
    class NeuralNetwork;
    class AdamOptimizer;
    class ReplayBuffer;
    struct PPOCoreConfig;
    class PPOCore;
}

class PPO : public godot::RefCounted {
    GDCLASS(PPO, godot::RefCounted);

private:
    std::unique_ptr<PPOInternal::PPOCore> ppo_core_;
    std::unique_ptr<PPOInternal::ReplayBuffer> replay_buffer_;
    
    Eigen::VectorXf current_observation_;
    int current_action_;
    float current_action_log_prob_;
    float current_value_estimate_;

    bool initialized_ = false;
    int observation_dim_;
    int action_dim_;
    int buffer_size_;
    int training_counter_ = 0;
    int train_every_n_steps_ = 2048;

    const uint32_t SAVE_FORMAT_VERSION = 1;

protected:
    static void _bind_methods();

public:
    PPO();
    ~PPO(); 

    void initialize(const godot::Dictionary& config);
    int get_action(const godot::PackedFloat32Array& observation_array); // Name bleibt, da "agent" nicht enthalten ist
    void store_experience(float reward, const godot::PackedFloat32Array& next_observation_array, bool done); // Name bleibt
    void train();

    bool save_model(const godot::String& file_path);
    bool load_model(const godot::String& file_path);

    // Eigen Konvertierungs-Helfer
    static Eigen::VectorXf packed_array_to_eigen(const godot::PackedFloat32Array& p_array);
    static godot::PackedFloat32Array eigen_to_packed_array(const Eigen::VectorXf& p_vector);
};

#endif