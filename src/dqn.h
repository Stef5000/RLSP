#ifndef DQN_H
#define DQN_H

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/classes/file_access.hpp>

using namespace godot;

// Represents a single step of experience
struct Transition {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

// --- NEW ---
// SumTree data structure for efficient priority sampling
class SumTree {
public:
    explicit SumTree(size_t capacity);

    void add(float priority, const Transition& transition);
    void update(size_t tree_idx, float priority);
    
    // Gets a leaf index, its priority, and the transition for a given value
    std::tuple<size_t, float, const Transition&> get_leaf(float value) const;
    
    float total_priority() const;
    size_t size() const;

private:
    void propagate(size_t idx, float change);

    size_t capacity;
    size_t write_idx;
    size_t current_size;
    std::vector<float> tree;
    std::vector<Transition> data;
};


// --- MODIFIED: ReplayBuffer now implements PER ---
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity = 100000, float alpha = 0.6f);

    void push(const Transition& transition);
    
    // Samples a batch, returning indices, transitions, and IS weights
    std::tuple<std::vector<size_t>, std::vector<const Transition*>, Eigen::VectorXf> sample(size_t batch_size, float beta, std::default_random_engine& gen);
    
    void update_priorities(const std::vector<size_t>& indices, const Eigen::VectorXf& errors);

    size_t size() const;

private:
    SumTree tree;
    float alpha;
    float max_priority;
    float per_epsilon;
};


// Represents the Neural Network itself
class DQN_Network {
public:
    DQN_Network();
    DQN_Network(int input_size, int hidden_size1, int hidden_size2, int output_size);

    Eigen::VectorXf predict(const Eigen::VectorXf& input) const;
    Eigen::MatrixXf predict_batch(const Eigen::MatrixXf& inputs) const;

    // --- MODIFIED: Train now accepts importance-sampling weights ---
    void train(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets, const Eigen::VectorXf& is_weights, float learning_rate);

    void soft_update_from(const DQN_Network& src, float tau);

    void save_to_file(const Ref<FileAccess>& file) const;
    void load_from_file(const Ref<FileAccess>& file);

    int input_size = 0;
    int hidden_size1 = 0;
    int hidden_size2 = 0;
    int output_size = 0;

private:
    Eigen::MatrixXf w1, w2, w3;
    Eigen::VectorXf b1, b2, b3;

    // Adam optimizer parameters
    Eigen::MatrixXf m_w1, v_w1;
    Eigen::MatrixXf m_w2, v_w2;
    Eigen::MatrixXf m_w3, v_w3;
    Eigen::VectorXf m_b1, v_b1;
    Eigen::VectorXf m_b2, v_b2;
    Eigen::VectorXf m_b3, v_b3;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    int adam_t = 0;
};


// The main Godot class, now named DQN
class DQN : public godot::RefCounted {
    GDCLASS(DQN, godot::RefCounted);

public:
    DQN();
    ~DQN() override = default;

    // --- MODIFIED: Added PER hyperparameters ---
    void initialize(int state_size, int action_size, float learning_rate, int batch_size,
                    float epsilon_decay, float epsilon_min, int hidden_size1, int hidden_size2,
                    float per_alpha = 0.6f, float per_beta_start = 0.4f, float per_beta_frames = 100000.0f);

    int get_action(const PackedFloat32Array& obs);
    void add_experience(const PackedFloat32Array& obs, int action, float reward,
                        const PackedFloat32Array& next_obs, bool done);
    void train();
    void end_episode();
    void save_model(const String &file_path);
    void load_model(const String &file_path);

protected:
    static void _bind_methods();

private:
    void update_target_network(); // Made private as it's an internal detail

    DQN_Network online_net;
    DQN_Network target_net;
    ReplayBuffer replay_buffer;

    float gamma;
    float epsilon;
    float epsilon_decay;
    float epsilon_min;
    float learning_rate;
    float tau;
    int input_size;
    int action_size;
    int batch_size;

    // --- NEW: PER parameters ---
    float per_alpha;
    float per_beta;
    float per_beta_start;
    float per_beta_increment;

    std::default_random_engine gen;
    int episode_count;
    int total_steps; // To anneal beta
};

#endif // DQN_H
