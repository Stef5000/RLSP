#include "dqn.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

using namespace godot;

// --- Helper functions for saving/loading Eigen types (unchanged) ---
namespace {
    // ... (Your save/load helper functions from the previous step go here)
    void save_eigen_matrix(const Ref<FileAccess>& file, const Eigen::MatrixXf& matrix) { PackedByteArray bytes; if (matrix.size() == 0) { file->store_buffer(bytes); return; } bytes.resize(matrix.size() * sizeof(float)); memcpy(bytes.ptrw(), matrix.data(), bytes.size()); file->store_buffer(bytes); }
    void load_eigen_matrix(const Ref<FileAccess>& file, Eigen::MatrixXf& matrix) { size_t expected_size = matrix.size() * sizeof(float); if (expected_size == 0) return; PackedByteArray bytes = file->get_buffer(expected_size); if (bytes.size() == expected_size) { memcpy(matrix.data(), bytes.ptr(), bytes.size()); } else { UtilityFunctions::print("Error: Mismatched matrix size on load."); } }
    void save_eigen_vector(const Ref<FileAccess>& file, const Eigen::VectorXf& vector) { PackedByteArray bytes; if (vector.size() == 0) { file->store_buffer(bytes); return; } bytes.resize(vector.size() * sizeof(float)); memcpy(bytes.ptrw(), vector.data(), bytes.size()); file->store_buffer(bytes); }
    void load_eigen_vector(const Ref<FileAccess>& file, Eigen::VectorXf& vector) { size_t expected_size = vector.size() * sizeof(float); if (expected_size == 0) return; PackedByteArray bytes = file->get_buffer(expected_size); if (bytes.size() == expected_size) { memcpy(vector.data(), bytes.ptr(), bytes.size()); } else { UtilityFunctions::print("Error: Mismatched vector size on load."); } }
}

// --- NEW: SumTree Implementation ---
SumTree::SumTree(size_t capacity) : capacity(capacity) {
    // Tree size is 2*capacity - 1
    tree.resize(2 * capacity - 1, 0.0f);
}

void SumTree::propagate(size_t tree_idx, float change) {
    size_t parent_idx = (tree_idx - 1) / 2;
    tree[parent_idx] += change;
    if (parent_idx != 0) {
        propagate(parent_idx, change);
    }
}

void SumTree::update(size_t tree_idx, float priority) {
    float change = priority - tree[tree_idx];
    tree[tree_idx] = priority;
    propagate(tree_idx, change);
}

void SumTree::add(float priority, size_t data_idx) {
    size_t tree_idx = data_idx + capacity - 1;
    update(tree_idx, priority);
}

float SumTree::total_priority() const {
    return tree[0];
}

std::tuple<size_t, float, size_t> SumTree::get_leaf(float value) const {
    size_t parent_idx = 0;
    while (true) {
        size_t left_child_idx = 2 * parent_idx + 1;
        size_t right_child_idx = left_child_idx + 1;
        
        // Reached a leaf node
        if (left_child_idx >= tree.size()) {
            break;
        }

        if (value <= tree[left_child_idx]) {
            parent_idx = left_child_idx;
        } else {
            value -= tree[left_child_idx];
            parent_idx = right_child_idx;
        }
    }
    size_t data_idx = parent_idx - (capacity - 1);
    return {parent_idx, tree[parent_idx], data_idx};
}

// --- NEW: PrioritizedReplayBuffer Implementation ---
PrioritizedReplayBuffer::PrioritizedReplayBuffer(size_t capacity, float alpha, float beta, float beta_increment)
    : tree(capacity), capacity(capacity), alpha(alpha), beta(beta), beta_increment(beta_increment),
      index(0), current_size(0), max_priority(1.0f), epsilon_per(1e-6f) {
    buffer.resize(capacity);
}

size_t PrioritizedReplayBuffer::size() const {
    return current_size;
}

void PrioritizedReplayBuffer::push(const Transition& transition) {
    buffer[index] = transition;
    // New experiences get max priority to ensure they are trained on
    tree.add(max_priority, index);

    index = (index + 1) % capacity;
    if (current_size < capacity) {
        current_size++;
    }
}

PER_SampleResult PrioritizedReplayBuffer::sample(size_t batch_size, std::default_random_engine& gen) {
    PER_SampleResult result;
    result.data_indices.reserve(batch_size);
    result.tree_indices.reserve(batch_size);
    result.is_weights.resize(batch_size);

    float total_p = tree.total_priority();
    float priority_segment = total_p / batch_size;
    
    // For stability, normalize weights by the maximum weight
    float max_weight = 0.0f;

    for (size_t i = 0; i < batch_size; ++i) {
        std::uniform_real_distribution<float> dist(priority_segment * i, priority_segment * (i + 1));
        float value = dist(gen);
        
        auto [tree_idx, priority, data_idx] = tree.get_leaf(value);

        result.data_indices.push_back(data_idx);
        result.tree_indices.push_back(tree_idx);
        
        float sampling_prob = priority / total_p;
        float weight = std::pow(current_size * sampling_prob, -beta);
        result.is_weights[i] = weight;

        if (weight > max_weight) {
            max_weight = weight;
        }
    }
    
    // Normalize weights
    if (max_weight > 0) {
        result.is_weights /= max_weight;
    }
    
    return result;
}

void PrioritizedReplayBuffer::update_priorities(const std::vector<size_t>& tree_indices, const Eigen::VectorXf& td_errors) {
    for (size_t i = 0; i < tree_indices.size(); ++i) {
        float priority = std::pow(td_errors[i] + epsilon_per, alpha);
        tree.update(tree_indices[i], priority);
        if (priority > max_priority) {
            max_priority = priority;
        }
    }
}

void PrioritizedReplayBuffer::anneal_beta() {
    beta = std::min(1.0f, beta + beta_increment);
}


// --- DQN_Network Implementation ---
// ... (Constructor, predict, predict_batch, soft_update_from, save_to_file, load_from_file are all fine from previous step)
// Constructor, predict, predict_batch, soft_update_from...
DQN_Network::DQN_Network() : input_size(0), hidden_size1(0), hidden_size2(0), output_size(0), adam_t(0) {}
DQN_Network::DQN_Network(int i, int h1, int h2, int o) : input_size(i), hidden_size1(h1), hidden_size2(h2), output_size(o), adam_t(0) { std::mt19937 r((unsigned)std::random_device{}()); std::normal_distribution<float> n(0.0f, 0.05f); if (i>0&&h1>0){w1=Eigen::MatrixXf::NullaryExpr(i,h1,[&](){return n(r);});b1=Eigen::VectorXf::Zero(h1);}if (h1>0&&h2>0){w2=Eigen::MatrixXf::NullaryExpr(h1,h2,[&](){return n(r);});b2=Eigen::VectorXf::Zero(h2);}if (h2>0&&o>0){w3=Eigen::MatrixXf::NullaryExpr(h2,o,[&](){return n(r);});b3=Eigen::VectorXf::Zero(o);}m_w1=Eigen::MatrixXf::Zero(std::max(1,i),std::max(1,h1));v_w1=Eigen::MatrixXf::Zero(std::max(1,i),std::max(1,h1));m_w2=Eigen::MatrixXf::Zero(std::max(1,h1),std::max(1,h2));v_w2=Eigen::MatrixXf::Zero(std::max(1,h1),std::max(1,h2));m_w3=Eigen::MatrixXf::Zero(std::max(1,h2),std::max(1,o));v_w3=Eigen::MatrixXf::Zero(std::max(1,h2),std::max(1,o));m_b1=Eigen::VectorXf::Zero(std::max(1,h1));v_b1=Eigen::VectorXf::Zero(std::max(1,h1));m_b2=Eigen::VectorXf::Zero(std::max(1,h2));v_b2=Eigen::VectorXf::Zero(std::max(1,h2));m_b3=Eigen::VectorXf::Zero(std::max(1,o));v_b3=Eigen::VectorXf::Zero(std::max(1,o)); }
Eigen::VectorXf DQN_Network::predict(const Eigen::VectorXf& input) const { Eigen::VectorXf L1=(w1.transpose()*input)+b1; for(int i=0;i<L1.size();++i)if(L1[i]<0.0f)L1[i]=0.0f; Eigen::VectorXf L2=(w2.transpose()*L1)+b2; for(int i=0;i<L2.size();++i)if(L2[i]<0.0f)L2[i]=0.0f; Eigen::VectorXf out=(w3.transpose()*L2)+b3; return out; }
Eigen::MatrixXf DQN_Network::predict_batch(const Eigen::MatrixXf& inputs) const { if(inputs.rows()==0)return Eigen::MatrixXf(); Eigen::MatrixXf L1=(inputs*w1).rowwise()+b1.transpose(); Eigen::MatrixXf A1=L1.unaryExpr([](float v){return v>0.0f?v:0.0f;}); Eigen::MatrixXf L2=(A1*w2).rowwise()+b2.transpose(); Eigen::MatrixXf A2=L2.unaryExpr([](float v){return v>0.0f?v:0.0f;}); Eigen::MatrixXf output=(A2*w3).rowwise()+b3.transpose(); return output; }
void DQN_Network::soft_update_from(const DQN_Network& src, float tau) { w1=tau*src.w1+(1.0f-tau)*w1;w2=tau*src.w2+(1.0f-tau)*w2;w3=tau*src.w3+(1.0f-tau)*w3;b1=tau*src.b1+(1.0f-tau)*b1;b2=tau*src.b2+(1.0f-tau)*b2;b3=tau*src.b3+(1.0f-tau)*b3; }
void DQN_Network::save_to_file(const Ref<FileAccess>& file) const { file->store_32(input_size); file->store_32(hidden_size1); file->store_32(hidden_size2); file->store_32(output_size); save_eigen_matrix(file, w1); save_eigen_matrix(file, w2); save_eigen_matrix(file, w3); save_eigen_vector(file, b1); save_eigen_vector(file, b2); save_eigen_vector(file, b3); file->store_32(adam_t); save_eigen_matrix(file, m_w1); save_eigen_matrix(file, v_w1); save_eigen_matrix(file, m_w2); save_eigen_matrix(file, v_w2); save_eigen_matrix(file, m_w3); save_eigen_matrix(file, v_w3); save_eigen_vector(file, m_b1); save_eigen_vector(file, v_b1); save_eigen_vector(file, m_b2); save_eigen_vector(file, v_b2); save_eigen_vector(file, m_b3); save_eigen_vector(file, v_b3); }
void DQN_Network::load_from_file(const Ref<FileAccess>& file) { int li=file->get_32();int lh1=file->get_32();int lh2=file->get_32();int lo=file->get_32(); if (li!=input_size||lh1!=hidden_size1||lh2!=hidden_size2||lo!=output_size){UtilityFunctions::print("Net arch mismatch, re-init.");*this=DQN_Network(li,lh1,lh2,lo);} load_eigen_matrix(file,w1);load_eigen_matrix(file,w2);load_eigen_matrix(file,w3);load_eigen_vector(file,b1);load_eigen_vector(file,b2);load_eigen_vector(file,b3); adam_t=file->get_32();load_eigen_matrix(file,m_w1);load_eigen_matrix(file,v_w1);load_eigen_matrix(file,m_w2);load_eigen_matrix(file,v_w2);load_eigen_matrix(file,m_w3);load_eigen_matrix(file,v_w3);load_eigen_vector(file,m_b1);load_eigen_vector(file,v_b1);load_eigen_vector(file,m_b2);load_eigen_vector(file,v_b2);load_eigen_vector(file,m_b3);load_eigen_vector(file,v_b3); }

// --- UPDATED train method to accept IS weights ---
void DQN_Network::train(const Eigen::MatrixXf& inputs, const Eigen::MatrixXf& targets, float learning_rate, const Eigen::VectorXf& is_weights) {
    const int m = static_cast<int>(inputs.rows());
    if (m == 0) return;

    Eigen::MatrixXf L1 = (inputs * w1).rowwise() + b1.transpose();
    Eigen::MatrixXf A1 = L1.unaryExpr([](float v){ return v > 0.0f ? v : 0.0f; });
    Eigen::MatrixXf L2 = (A1 * w2).rowwise() + b2.transpose();
    Eigen::MatrixXf A2 = L2.unaryExpr([](float v){ return v > 0.0f ? v : 0.0f; });
    Eigen::MatrixXf predictions = (A2 * w3).rowwise() + b3.transpose();

    Eigen::MatrixXf error = predictions - targets;

    // --- PER CHANGE: Scale the error by the IS weights ---
    for (int i = 0; i < m; ++i) {
        error.row(i) *= is_weights[i];
    }
    
    Eigen::MatrixXf dPred = (2.0f / m) * error;
    // ... (rest of backpropagation is unchanged)
    Eigen::MatrixXf dW3 = A2.transpose() * dPred; Eigen::VectorXf dB3 = dPred.colwise().sum();
    Eigen::MatrixXf dA2 = dPred * w3.transpose(); Eigen::MatrixXf dL2 = dA2.array() * (L2.array() > 0.0f).cast<float>(); Eigen::MatrixXf dW2 = A1.transpose() * dL2; Eigen::VectorXf dB2 = dL2.colwise().sum();
    Eigen::MatrixXf dA1 = dL2 * w2.transpose(); Eigen::MatrixXf dL1 = dA1.array() * (L1.array() > 0.0f).cast<float>(); Eigen::MatrixXf dW1 = inputs.transpose() * dL1; Eigen::VectorXf dB1 = dL1.colwise().sum();
    auto clip = [](Eigen::Ref<Eigen::MatrixXf> G, float thr){float n=G.norm();if(n>thr)G*=(thr/n);};clip(dW1,10.0f);clip(dW2,10.0f);clip(dW3,10.0f);
    adam_t++;
    auto adam_step_matrix = [this, learning_rate](Eigen::MatrixXf&p,Eigen::MatrixXf&mp,Eigen::MatrixXf&vp,const Eigen::MatrixXf&g){mp=beta1*mp+(1.0f-beta1)*g;vp=beta2*vp+(1.0f-beta2)*g.array().square().matrix();Eigen::MatrixXf mh=mp/(1.0f-std::pow(beta1,adam_t));Eigen::MatrixXf vh=vp/(1.0f-std::pow(beta2,adam_t));p.noalias()-=learning_rate*(mh.array()/(vh.array().sqrt()+adam_epsilon)).matrix();};
    auto adam_step_vector = [this, learning_rate](Eigen::VectorXf&p,Eigen::VectorXf&mp,Eigen::VectorXf&vp,const Eigen::VectorXf&g){mp=beta1*mp+(1.0f-beta1)*g;vp=beta2*vp+(1.0f-beta2)*g.array().square().matrix();Eigen::VectorXf mh=mp/(1.0f-std::pow(beta1,adam_t));Eigen::VectorXf vh=vp/(1.0f-std::pow(beta2,adam_t));p.noalias()-=learning_rate*(mh.array()/(vh.array().sqrt()+adam_epsilon)).matrix();};
    adam_step_matrix(w3,m_w3,v_w3,dW3);adam_step_vector(b3,m_b3,v_b3,dB3);adam_step_matrix(w2,m_w2,v_w2,dW2);adam_step_vector(b2,m_b2,v_b2,dB2);adam_step_matrix(w1,m_w1,v_w1,dW1);adam_step_vector(b1,m_b1,v_b1,dB1);
}

// ---------------------- DQN (Godot Class) Implementation ----------------------

void DQN::_bind_methods() {
    // UPDATED initialize signature
    ClassDB::bind_method(D_METHOD("initialize", "state_size", "action_size", "learning_rate", "batch_size", "epsilon_decay", "epsilon_min", "hidden_size1", "hidden_size2", "per_alpha", "per_beta_start", "per_beta_increment"), &DQN::initialize);
    ClassDB::bind_method(D_METHOD("get_action", "obs"), &DQN::get_action);
    ClassDB::bind_method(D_METHOD("add_experience", "obs", "action", "reward", "next_obs", "done"), &DQN::add_experience);
    ClassDB::bind_method(D_METHOD("train"), &DQN::train);
    ClassDB::bind_method(D_METHOD("update_target_network"), &DQN::update_target_network);
    ClassDB::bind_method(D_METHOD("end_episode"), &DQN::end_episode);
    ClassDB::bind_method(D_METHOD("save_model", "file_path"), &DQN::save_model);
    ClassDB::bind_method(D_METHOD("load_model", "file_path"), &DQN::load_model);
}

DQN::DQN() :
    replay_buffer(), // Default constructor is fine
    gamma(0.99f),
    epsilon(1.0f),
    epsilon_decay(0.9995f),
    epsilon_min(0.05f),
    learning_rate(0.001f),
    tau(0.01f),
    input_size(0),
    action_size(0),
    batch_size(64),
    gen(std::random_device{}()),
    episode_count(0)
{
}

// UPDATED initialize method
void DQN::initialize(int state_size, int action_size, float p_learning_rate, int p_batch_size, 
                     float p_epsilon_decay, float p_epsilon_min, int p_hidden_size1, int p_hidden_size2,
                     float per_alpha, float per_beta_start, float per_beta_increment) {
    this->input_size = state_size;
    this->action_size = action_size;
    learning_rate = p_learning_rate;
    batch_size = p_batch_size;
    epsilon_decay = p_epsilon_decay;
    epsilon_min = p_epsilon_min;

    // Initialize PER buffer with new params
    replay_buffer = PrioritizedReplayBuffer(50000, per_alpha, per_beta_start, per_beta_increment);

    online_net = DQN_Network(state_size, p_hidden_size1, p_hidden_size2, action_size);
    target_net = online_net;
}

// get_action and add_experience are unchanged
int DQN::get_action(const PackedFloat32Array& obs) { std::uniform_real_distribution<float> dist(0.0f,1.0f); if(dist(gen)<epsilon){std::uniform_int_distribution<int> action_dist(0,action_size-1);return action_dist(gen);} Eigen::VectorXf input(input_size); for(int i=0;i<input_size;++i)input[i]=obs[i]; Eigen::VectorXf q_values=online_net.predict(input); Eigen::Index max_index=0; q_values.maxCoeff(&max_index); return static_cast<int>(max_index); }
void DQN::add_experience(const PackedFloat32Array& obs, int action, float reward, const PackedFloat32Array& next_obs, bool done) { Transition t; t.state.resize(obs.size());t.next_state.resize(next_obs.size()); for(int i=0;i<obs.size();++i)t.state[i]=obs[i]; for(int i=0;i<next_obs.size();++i)t.next_state[i]=next_obs[i]; t.action=action;t.reward=reward;t.done=done; replay_buffer.push(t); }

// --- COMPLETELY REWRITTEN train method ---
void DQN::train() {
    if (replay_buffer.size() < static_cast<size_t>(batch_size)) return;

    // 1. Sample from the PER buffer
    PER_SampleResult sample = replay_buffer.sample(batch_size, gen);
    if (sample.data_indices.empty()) return;

    Eigen::MatrixXf states(batch_size, input_size);
    Eigen::MatrixXf next_states(batch_size, input_size);
    std::vector<int> actions(batch_size);
    Eigen::VectorXf rewards(batch_size);
    std::vector<char> dones(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        // We need to access the original buffer using the sampled indices
        const Transition& t = replay_buffer.buffer[sample.data_indices[i]]; // IMPORTANT: Access internal buffer
        states.row(i) = Eigen::Map<const Eigen::RowVectorXf>(t.state.data(), input_size);
        next_states.row(i) = Eigen::Map<const Eigen::RowVectorXf>(t.next_state.data(), input_size);
        actions[i] = t.action;
        rewards[i] = t.reward;
        dones[i] = t.done ? 1 : 0;
    }

    // 2. Compute targets (Double DQN logic is unchanged)
    Eigen::MatrixXf q_states = online_net.predict_batch(states);
    Eigen::MatrixXf q_next_online = online_net.predict_batch(next_states);
    Eigen::MatrixXf q_next_target = target_net.predict_batch(next_states);
    Eigen::MatrixXf targets = q_states;

    for (int i = 0; i < batch_size; ++i) {
        float target_value = rewards[i];
        if (!dones[i]) {
            Eigen::Index best_action = 0;
            q_next_online.row(i).maxCoeff(&best_action);
            target_value += gamma * q_next_target(i, best_action);
        }
        targets(i, actions[i]) = target_value;
    }

    // 3. Calculate TD-errors to update priorities
    Eigen::VectorXf td_errors(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        td_errors[i] = std::abs(targets(i, actions[i]) - q_states(i, actions[i]));
    }
    replay_buffer.update_priorities(sample.tree_indices, td_errors);

    // 4. Train the network, passing the IS weights
    online_net.train(states, targets, learning_rate, sample.is_weights);
    
    // 5. Anneal beta and update target network
    replay_buffer.anneal_beta();
    update_target_network();
}

void DQN::update_target_network() {
    target_net.soft_update_from(online_net, tau);
}

void DQN::end_episode() {
    episode_count++;
    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
}

// --- UPDATED save/load to include PER beta state ---
void DQN::save_model(const String &file_path) {
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE);
    if (file.is_null()) {
        UtilityFunctions::print("Error: Could not save DQN model to ", file_path);
        return;
    }
    online_net.save_to_file(file);
    file->store_float(epsilon);
    file->store_32(episode_count);
    file->store_float(replay_buffer.get_beta()); // Save current beta
    file->close();
    UtilityFunctions::print("Successfully saved model to ", file_path);
}

void DQN::load_model(const String &file_path) {
    if (!FileAccess::file_exists(file_path)) {
        UtilityFunctions::print("Error: File not found, cannot load from ", file_path);
        return;
    }
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if (file.is_null()) { return; }

    online_net.load_from_file(file);

    if (!file->eof_reached()) {
        epsilon = file->get_float();
        episode_count = file->get_32();
    } else {
        epsilon = epsilon_min; episode_count = 0;
        UtilityFunctions::print("Warning: Loading older model format. Epsilon/episode count not found.");
    }
    
    if (!file->eof_reached()) {
        float loaded_beta = file->get_float(); // Load beta
        // This requires re-initializing the buffer with the loaded beta
        replay_buffer = PrioritizedReplayBuffer(replay_buffer.capacity, replay_buffer.alpha, loaded_beta, replay_buffer.beta_increment);
    }

    target_net = online_net;
    file->close();
    UtilityFunctions::print("Successfully loaded model from ", file_path, ". Epsilon: ", epsilon);
}
