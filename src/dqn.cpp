#include "dqn.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

using namespace godot;

// --- ReplayBuffer Implementation (unchanged) ---
ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity(capacity), index(0), current_size(0) {
    buffer.resize(capacity);
}

void ReplayBuffer::push(const Transition& transition) {
    buffer[index] = transition;
    index = (index + 1) % capacity;
    if (current_size < capacity) {
        current_size++;
    }
}

std::vector<size_t> ReplayBuffer::sample_indices(size_t batch_size, std::default_random_engine& gen) const {
    std::vector<size_t> indices;
    if (current_size == 0) return indices;
    indices.reserve(batch_size);
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        indices.push_back(dist(gen));
    }
    return indices;
}

const Transition& ReplayBuffer::get(size_t idx) const {
    return buffer[idx];
}

size_t ReplayBuffer::size() const {
    return current_size;
}

// --- DQN_Network Implementation (save/load methods updated) ---

DQN_Network::DQN_Network() : input_size(0), hidden_size1(0), hidden_size2(0), output_size(0), adam_t(0) {}

DQN_Network::DQN_Network(int input_size_, int hidden_size1_, int hidden_size2_, int output_size_)
    : input_size(input_size_), hidden_size1(hidden_size1_), hidden_size2(hidden_size2_),
      output_size(output_size_), adam_t(0)
{
    std::mt19937 rng((unsigned)std::random_device{}());
    std::normal_distribution<float> nd(0.0f, 0.05f);

    if (input_size > 0 && hidden_size1 > 0) {
        w1 = Eigen::MatrixXf::NullaryExpr(input_size, hidden_size1, [&](){ return nd(rng); });
        b1 = Eigen::VectorXf::Zero(hidden_size1);
    }
    if (hidden_size1 > 0 && hidden_size2 > 0) {
        w2 = Eigen::MatrixXf::NullaryExpr(hidden_size1, hidden_size2, [&](){ return nd(rng); });
        b2 = Eigen::VectorXf::Zero(hidden_size2);
    }
    if (hidden_size2 > 0 && output_size > 0) {
        w3 = Eigen::MatrixXf::NullaryExpr(hidden_size2, output_size, [&](){ return nd(rng); });
        b3 = Eigen::VectorXf::Zero(output_size);
    }

    m_w1 = Eigen::MatrixXf::Zero(std::max(1, input_size), std::max(1, hidden_size1));
    v_w1 = Eigen::MatrixXf::Zero(std::max(1, input_size), std::max(1, hidden_size1));
    m_w2 = Eigen::MatrixXf::Zero(std::max(1, hidden_size1), std::max(1, hidden_size2));
    v_w2 = Eigen::MatrixXf::Zero(std::max(1, hidden_size1), std::max(1, hidden_size2));
    m_w3 = Eigen::MatrixXf::Zero(std::max(1, hidden_size2), std::max(1, output_size));
    v_w3 = Eigen::MatrixXf::Zero(std::max(1, hidden_size2), std::max(1, output_size));

    m_b1 = Eigen::VectorXf::Zero(std::max(1, hidden_size1));
    v_b1 = Eigen::VectorXf::Zero(std::max(1, hidden_size1));
    m_b2 = Eigen::VectorXf::Zero(std::max(1, hidden_size2));
    v_b2 = Eigen::VectorXf::Zero(std::max(1, hidden_size2));
    m_b3 = Eigen::VectorXf::Zero(std::max(1, output_size));
    v_b3 = Eigen::VectorXf::Zero(std::max(1, output_size));
}

Eigen::VectorXf DQN_Network::predict(const Eigen::VectorXf& input) const {
    Eigen::VectorXf L1 = (w1.transpose() * input) + b1;
    for (int i = 0; i < L1.size(); ++i) if (L1[i] < 0.0f) L1[i] = 0.0f;
    Eigen::VectorXf L2 = (w2.transpose() * L1) + b2;
    for (int i = 0; i < L2.size(); ++i) if (L2[i] < 0.0f) L2[i] = 0.0f;
    Eigen::VectorXf out = (w3.transpose() * L2) + b3;
    return out;
}

Eigen::MatrixXf DQN_Network::predict_batch(const Eigen::MatrixXf& inputs) const {
    if (inputs.rows() == 0) return Eigen::MatrixXf();
    Eigen::MatrixXf L1 = (inputs * w1).rowwise() + b1.transpose();
    Eigen::MatrixXf A1 = L1.unaryExpr([](float v){ return v > 0.0f ? v : 0.0f; });
    Eigen::MatrixXf L2 = (A1 * w2).rowwise() + b2.transpose();
    Eigen::MatrixXf A2 = L2.unaryExpr([](float v){ return v > 0.0f ? v : 0.0f; });
    Eigen::MatrixXf output = (A2 * w3).rowwise() + b3.transpose();
    return output;
}

void DQN_Network::train(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets, float learning_rate) {
    const int m = static_cast<int>(inputs.rows());
    if (m == 0) return;

    Eigen::MatrixXf L1 = (inputs * w1).rowwise() + b1.transpose();
    Eigen::MatrixXf A1 = L1.unaryExpr([](float v){ return v > 0.0f ? v : 0.0f; });
    Eigen::MatrixXf L2 = (A1 * w2).rowwise() + b2.transpose();
    Eigen::MatrixXf A2 = L2.unaryExpr([](float v){ return v > 0.0f ? v : 0.0f; });
    Eigen::MatrixXf predictions = (A2 * w3).rowwise() + b3.transpose();

    Eigen::MatrixXf error = predictions - targets;
    Eigen::MatrixXf dPred = (2.0f / m) * error;

    Eigen::MatrixXf dW3 = A2.transpose() * dPred;
    Eigen::VectorXf dB3 = dPred.colwise().sum();

    Eigen::MatrixXf dA2 = dPred * w3.transpose();
    Eigen::MatrixXf dL2 = dA2.array() * (L2.array() > 0.0f).cast<float>();
    Eigen::MatrixXf dW2 = A1.transpose() * dL2;
    Eigen::VectorXf dB2 = dL2.colwise().sum();

    Eigen::MatrixXf dA1 = dL2 * w2.transpose();
    Eigen::MatrixXf dL1 = dA1.array() * (L1.array() > 0.0f).cast<float>();
    Eigen::MatrixXf dW1 = inputs.transpose() * dL1;
    Eigen::VectorXf dB1 = dL1.colwise().sum();

    auto clip = [](Eigen::Ref<Eigen::MatrixXf> G, float thr){
        float n = G.norm();
        if (n > thr) G *= (thr / n);
    };
    const float clip_thr = 10.0f;
    clip(dW1, clip_thr); clip(dW2, clip_thr); clip(dW3, clip_thr);

    adam_t++;
    auto adam_step_matrix = [this, learning_rate](Eigen::MatrixXf& param, Eigen::MatrixXf& m_param, Eigen::MatrixXf& v_param, const Eigen::MatrixXf& grad) {
        m_param = beta1 * m_param + (1.0f - beta1) * grad;
        v_param = beta2 * v_param + (1.0f - beta2) * grad.array().square().matrix();
        Eigen::MatrixXf m_hat = m_param / (1.0f - std::pow(beta1, adam_t));
        Eigen::MatrixXf v_hat = v_param / (1.0f - std::pow(beta2, adam_t));
        param.noalias() -= learning_rate * (m_hat.array() / (v_hat.array().sqrt() + adam_epsilon)).matrix();
    };

    auto adam_step_vector = [this, learning_rate](Eigen::VectorXf& param, Eigen::VectorXf& m_param, Eigen::VectorXf& v_param, const Eigen::VectorXf& grad) {
        m_param = beta1 * m_param + (1.0f - beta1) * grad;
        v_param = beta2 * v_param + (1.0f - beta2) * grad.array().square().matrix();
        Eigen::VectorXf m_hat = m_param / (1.0f - std::pow(beta1, adam_t));
        Eigen::VectorXf v_hat = v_param / (1.0f - std::pow(beta2, adam_t));
        param.noalias() -= learning_rate * (m_hat.array() / (v_hat.array().sqrt() + adam_epsilon)).matrix();
    };

    adam_step_matrix(w3, m_w3, v_w3, dW3);
    adam_step_vector(b3, m_b3, v_b3, dB3);
    adam_step_matrix(w2, m_w2, v_w2, dW2);
    adam_step_vector(b2, m_b2, v_b2, dB2);
    adam_step_matrix(w1, m_w1, v_w1, dW1);
    adam_step_vector(b1, m_b1, v_b1, dB1);
}

void DQN_Network::soft_update_from(const DQN_Network& src, float tau) {
    w1 = tau * src.w1 + (1.0f - tau) * w1;
    w2 = tau * src.w2 + (1.0f - tau) * w2;
    w3 = tau * src.w3 + (1.0f - tau) * w3;
    b1 = tau * src.b1 + (1.0f - tau) * b1;
    b2 = tau * src.b2 + (1.0f - tau) * b2;
    b3 = tau * src.b3 + (1.0f - tau) * b3;
}

void DQN_Network::save_to_file(const Ref<FileAccess>& file) const {
    file->store_32(input_size);
    file->store_32(hidden_size1);
    file->store_32(hidden_size2);
    file->store_32(output_size);

    PackedByteArray w1_bytes; w1_bytes.resize(w1.size() * sizeof(float)); memcpy(w1_bytes.ptrw(), w1.data(), w1_bytes.size()); file->store_buffer(w1_bytes);
    PackedByteArray w2_bytes; w2_bytes.resize(w2.size() * sizeof(float)); memcpy(w2_bytes.ptrw(), w2.data(), w2_bytes.size()); file->store_buffer(w2_bytes);
    PackedByteArray w3_bytes; w3_bytes.resize(w3.size() * sizeof(float)); memcpy(w3_bytes.ptrw(), w3.data(), w3_bytes.size()); file->store_buffer(w3_bytes);
    PackedByteArray b1_bytes; b1_bytes.resize(b1.size() * sizeof(float)); memcpy(b1_bytes.ptrw(), b1.data(), b1_bytes.size()); file->store_buffer(b1_bytes);
    PackedByteArray b2_bytes; b2_bytes.resize(b2.size() * sizeof(float)); memcpy(b2_bytes.ptrw(), b2.data(), b2_bytes.size()); file->store_buffer(b2_bytes);
    PackedByteArray b3_bytes; b3_bytes.resize(b3.size() * sizeof(float)); memcpy(b3_bytes.ptrw(), b3.data(), b3_bytes.size()); file->store_buffer(b3_bytes);
}

void DQN_Network::load_from_file(const Ref<FileAccess>& file) {
    int li = file->get_32();
    int lh1 = file->get_32();
    int lh2 = file->get_32();
    int lo = file->get_32();

    if (li != input_size || lh1 != hidden_size1 || lh2 != hidden_size2 || lo != output_size) {
        input_size = li; hidden_size1 = lh1; hidden_size2 = lh2; output_size = lo;
        w1.resize(input_size, hidden_size1); w2.resize(hidden_size1, hidden_size2); w3.resize(hidden_size2, output_size);
        b1.resize(hidden_size1); b2.resize(hidden_size2); b3.resize(output_size);
    }
    
    PackedByteArray w1_bytes = file->get_buffer(w1.size() * sizeof(float)); memcpy(w1.data(), w1_bytes.ptr(), w1_bytes.size());
    PackedByteArray w2_bytes = file->get_buffer(w2.size() * sizeof(float)); memcpy(w2.data(), w2_bytes.ptr(), w2_bytes.size());
    PackedByteArray w3_bytes = file->get_buffer(w3.size() * sizeof(float)); memcpy(w3.data(), w3_bytes.ptr(), w3_bytes.size());
    PackedByteArray b1_bytes = file->get_buffer(b1.size() * sizeof(float)); memcpy(b1.data(), b1_bytes.ptr(), b1_bytes.size());
    PackedByteArray b2_bytes = file->get_buffer(b2.size() * sizeof(float)); memcpy(b2.data(), b2_bytes.ptr(), b2_bytes.size());
    PackedByteArray b3_bytes = file->get_buffer(b3.size() * sizeof(float)); memcpy(b3.data(), b3_bytes.ptr(), b3_bytes.size());
}


// ---------------------- DQN (Godot Class) Implementation ----------------------

void DQN::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "state_size", "action_size", "learning_rate", "batch_size", "epsilon_decay", "epsilon_min", "hidden_size1", "hidden_size2"), &DQN::initialize);
    ClassDB::bind_method(D_METHOD("get_action", "obs"), &DQN::get_action);
    ClassDB::bind_method(D_METHOD("add_experience", "obs", "action", "reward", "next_obs", "done"), &DQN::add_experience);
    ClassDB::bind_method(D_METHOD("train"), &DQN::train);
    ClassDB::bind_method(D_METHOD("update_target_network"), &DQN::update_target_network);
    ClassDB::bind_method(D_METHOD("end_episode"), &DQN::end_episode);
    ClassDB::bind_method(D_METHOD("save_model", "file_path"), &DQN::save_model);
    ClassDB::bind_method(D_METHOD("load_model", "file_path"), &DQN::load_model);
}

DQN::DQN() :
    online_net(),
    target_net(),
    replay_buffer(50000),
    batch_size(64),
    gamma(0.99f),
    epsilon(1.0f),
    epsilon_decay(0.9995f),
    epsilon_min(0.05f),
    learning_rate(0.001f),
    tau(0.01f),
    input_size(0),
    action_size(0),
    gen(std::random_device{}()),
    episode_count(0)
{
    // Removed non-portable thread setup
}

void DQN::initialize(int state_size, int action_size, float p_learning_rate, int p_batch_size, float p_epsilon_decay, float p_epsilon_min, int p_hidden_size1, int p_hidden_size2) {
    this->input_size = state_size;
    this->action_size = action_size;
    learning_rate = p_learning_rate;
    batch_size = p_batch_size;
    epsilon_decay = p_epsilon_decay;
    epsilon_min = p_epsilon_min;

    online_net = DQN_Network(state_size, p_hidden_size1, p_hidden_size2, action_size);
    target_net = online_net;
}

int DQN::get_action(const PackedFloat32Array& obs) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(gen) < epsilon) {
        std::uniform_int_distribution<int> action_dist(0, action_size - 1);
        return action_dist(gen);
    }
    Eigen::VectorXf input(input_size);
    for (int i = 0; i < input_size; ++i) input[i] = obs[i];
    Eigen::VectorXf q_values = online_net.predict(input);
    Eigen::Index max_index = 0;
    q_values.maxCoeff(&max_index);
    return static_cast<int>(max_index);
}

void DQN::add_experience(const PackedFloat32Array& obs, int action, float reward, const PackedFloat32Array& next_obs, bool done) {
    Transition t;
    t.state.resize(obs.size());
    t.next_state.resize(next_obs.size());
    for (int i = 0; i < obs.size(); ++i) t.state[i] = obs[i];
    for (int i = 0; i < next_obs.size(); ++i) t.next_state[i] = next_obs[i];
    t.action = action;
    t.reward = reward;
    t.done = done;
    replay_buffer.push(t);
}

void DQN::train() {
    if (replay_buffer.size() < static_cast<size_t>(batch_size)) return;

    auto indices = replay_buffer.sample_indices(batch_size, gen);
    if (indices.empty()) return;

    Eigen::MatrixXf states(batch_size, input_size);
    Eigen::MatrixXf next_states(batch_size, input_size);
    Eigen::MatrixXf targets(batch_size, action_size);

    std::vector<int> actions(batch_size);
    Eigen::VectorXf rewards(batch_size);
    std::vector<char> dones(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const Transition& t = replay_buffer.get(indices[i]);
        states.row(i) = Eigen::Map<const Eigen::RowVectorXf>(t.state.data(), input_size);
        next_states.row(i) = Eigen::Map<const Eigen::RowVectorXf>(t.next_state.data(), input_size);
        actions[i] = t.action;
        rewards[i] = t.reward;
        dones[i] = t.done ? 1 : 0;
    }

    Eigen::MatrixXf q_states = online_net.predict_batch(states);
    Eigen::MatrixXf q_next_online = online_net.predict_batch(next_states);
    Eigen::MatrixXf q_next_target = target_net.predict_batch(next_states);

    targets = q_states;

    for (int i = 0; i < batch_size; ++i) {
        float target_value = rewards[i];
        if (!dones[i]) {
            Eigen::Index best_action = 0;
            q_next_online.row(i).maxCoeff(&best_action);
            target_value += gamma * q_next_target(i, best_action);
        }
        targets(i, actions[i]) = target_value;
    }

    online_net.train(states, targets, learning_rate);

    update_target_network();
}

void DQN::update_target_network() {
    target_net.soft_update_from(online_net, tau);
}

void DQN::end_episode() {
    episode_count++;
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
}

void DQN::save_model(const String &file_path) {
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE);
    if (file.is_null()) {
        UtilityFunctions::print("Error: Could not save DQN model to ", file_path);
        return;
    }
    online_net.save_to_file(file);
    file->close();
}

void DQN::load_model(const String &file_path) {
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if (file.is_null()) {
        UtilityFunctions::print("Error: Could not load DQN model from ", file_path);
        return;
    }
    online_net.load_from_file(file);
    target_net = online_net; // Sync target network
    file->close();
}
