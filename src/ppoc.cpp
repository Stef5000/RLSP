#define _USE_MATH_DEFINES
#include "ppoc.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/file_access.hpp>

#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>


#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

const uint32_t PPO_AGENT_CONTINUOUS_MAGIC_NUMBER = 0x50504F43;
const uint32_t PPO_AGENT_CONTINUOUS_FORMAT_VERSION = 1;

// --- Start of PPOInternalContinuous and Helper functions (unchanged) ---
// This large section remains the same as your provided code.
// To keep the response concise, I'm omitting the full text,
// which you can copy from your original file. It has no changes.
// --- (Eigen <-> PackedArray helpers and PPOInternalContinuous namespace) ---
// ...
// --- End of PPOInternalContinuous ---
// Helper: Eigen <-> Packed...(Array) ---
Eigen::VectorXf PPOC::packed_array_to_eigen_vector(const godot::PackedFloat32Array& p_array) {
    Eigen::VectorXf vec(p_array.size());
    for (int i = 0; i < p_array.size(); ++i) {
        vec[i] = p_array[i];
    }
    return vec;
}
godot::PackedFloat32Array PPOC::eigen_vector_to_packed_array(const Eigen::VectorXf& p_vector) {
    godot::PackedFloat32Array arr;
    arr.resize(p_vector.size());
    for (int i = 0; i < p_vector.size(); ++i) {
        arr[i] = p_vector[i];
    }
    return arr;
}
godot::PackedByteArray PPOC::eigen_vector_to_byte_array(const Eigen::VectorXf& vec) {
    godot::PackedByteArray byte_array;
    if (vec.size() == 0) return byte_array;
    byte_array.resize(vec.size() * sizeof(float));
    memcpy(byte_array.ptrw(), vec.data(), vec.size() * sizeof(float));
    return byte_array;
}
Eigen::VectorXf PPOC::byte_array_to_eigen_vector(const godot::PackedByteArray& byte_arr, int size) {
    if (size < 0) {
        return Eigen::VectorXf();
    }
    if (byte_arr.size() != static_cast<uint64_t>(size) * sizeof(float)) {
        return Eigen::VectorXf::Zero(size);
    }
    if (size == 0) return Eigen::VectorXf();
    Eigen::VectorXf vec(size);
    memcpy(vec.data(), byte_arr.ptr(), static_cast<uint64_t>(size) * sizeof(float));
    return vec;
}
godot::PackedByteArray PPOC::eigen_matrix_to_byte_array(const Eigen::MatrixXf& mat) {
    godot::PackedByteArray byte_array;
    if (mat.size() == 0) return byte_array;
    byte_array.resize(mat.size() * sizeof(float));
    memcpy(byte_array.ptrw(), mat.data(), mat.size() * sizeof(float));
    return byte_array;
}
Eigen::MatrixXf PPOC::byte_array_to_eigen_matrix(const godot::PackedByteArray& byte_arr, int rows, int cols) {
    if (rows < 0 || cols < 0) {
        return Eigen::MatrixXf();
    }
    if (byte_arr.size() != static_cast<uint64_t>(rows) * cols * sizeof(float)) {
        return Eigen::MatrixXf::Zero(rows, cols);
    }
    if (rows == 0 || cols == 0) return Eigen::MatrixXf(rows,cols);
    Eigen::MatrixXf mat(rows, cols);
    memcpy(mat.data(), byte_arr.ptr(), static_cast<uint64_t>(rows) * cols * sizeof(float));
    return mat;
}


namespace PPOInternalContinuous {

enum class ActivationType { RELU, TANH, LINEAR, SOFTMAX };

Eigen::VectorXf apply_activation(const Eigen::VectorXf& x, ActivationType type) {
    switch (type) {
        case ActivationType::RELU: return x.unaryExpr([](float v){ return std::max(0.0f, v); });
        case ActivationType::TANH: return x.unaryExpr([](float v){ return std::tanh(v); });
        case ActivationType::LINEAR: return x;
        case ActivationType::SOFTMAX: {
            if (x.size() == 0) return x;
            Eigen::VectorXf exp_x = (x.array() - x.maxCoeff()).exp();
            float sum_exp_x = exp_x.sum();
            if (std::abs(sum_exp_x) < 1e-8f) return Eigen::VectorXf::Constant(x.size(), 1.0f/static_cast<float>(x.size()));
            return exp_x / sum_exp_x;
        }
        default: return x;
    }
}

Eigen::MatrixXf activation_derivative(const Eigen::VectorXf& activated_output, ActivationType type) {
    Eigen::VectorXf derivatives;
    if (activated_output.size() == 0) return Eigen::MatrixXf(0,0);
    switch (type) {
        case ActivationType::RELU: derivatives = activated_output.unaryExpr([](float v){ return v > 0.0f ? 1.0f : 0.0f; }); break;
        case ActivationType::TANH: derivatives = activated_output.unaryExpr([](float v){ return 1.0f - v * v; }); break;
        case ActivationType::LINEAR: derivatives = Eigen::VectorXf::Ones(activated_output.size()); break;
        case ActivationType::SOFTMAX: derivatives = Eigen::VectorXf::Ones(activated_output.size()); break;
        default: derivatives = Eigen::VectorXf::Ones(activated_output.size());
    }
    return derivatives.asDiagonal();
}

class NeuralNetworkContinuous {
public:
    std::vector<Eigen::MatrixXf> weights_; std::vector<Eigen::VectorXf> biases_; std::vector<ActivationType> layer_activations_;
    std::vector<Eigen::VectorXf> layer_outputs_pre_activation_; std::vector<Eigen::VectorXf> layer_outputs_activated_;
    std::mt19937 gen; std::vector<Eigen::MatrixXf> grad_weights_; std::vector<Eigen::VectorXf> grad_biases_;

    NeuralNetworkContinuous(int input_dim, const std::vector<int>& hidden_dims, int output_dim, 
                  ActivationType hidden_activation, ActivationType output_activation, unsigned int seed = std::random_device{}()) 
                  : gen(seed) {
        int current_dim = input_dim;
        for (size_t i = 0; i < hidden_dims.size(); ++i) {
            add_layer(current_dim, hidden_dims[i], hidden_activation);
            current_dim = hidden_dims[i];
        }
        add_layer(current_dim, output_dim, output_activation);
    }

    void add_layer(int input_dim, int output_dim, ActivationType activation) {
        if (input_dim <=0 || output_dim <=0) {return;}
        float limit = std::sqrt(6.0f / static_cast<float>(input_dim + output_dim));
        std::uniform_real_distribution<float> dist(-limit, limit);
        Eigen::MatrixXf W(output_dim, input_dim);
        for(int r=0; r<W.rows(); ++r) for(int c=0; c<W.cols(); ++c) W(r,c) = dist(gen);
        weights_.push_back(W);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(output_dim); 
        biases_.push_back(b);
        layer_activations_.push_back(activation);
    }

    Eigen::VectorXf forward(const Eigen::VectorXf& input) {
        layer_outputs_pre_activation_.clear(); layer_outputs_activated_.clear();
        if (input.size() == 0 && !weights_.empty() && weights_[0].cols() != 0) { 
             return Eigen::VectorXf();
        }
        Eigen::VectorXf current_output = input;
        layer_outputs_activated_.push_back(current_output); 
        for (size_t i = 0; i < weights_.size(); ++i) {
            if (weights_[i].cols() != current_output.size()) {
                return Eigen::VectorXf();
            }
            Eigen::VectorXf pre_activation = weights_[i] * current_output + biases_[i];
            layer_outputs_pre_activation_.push_back(pre_activation);
            current_output = apply_activation(pre_activation, layer_activations_[i]);
            layer_outputs_activated_.push_back(current_output);
        }
        return current_output;
    }
    
    void backward(const Eigen::VectorXf& loss_gradient_wrt_output) {
        if (weights_.empty() || loss_gradient_wrt_output.size() == 0 || layer_outputs_activated_.empty()) return;
        if (layer_outputs_activated_.back().size() != loss_gradient_wrt_output.size()){
            return;
        }

        grad_weights_.assign(weights_.size(), Eigen::MatrixXf());
        grad_biases_.assign(biases_.size(), Eigen::VectorXf());
        Eigen::VectorXf current_delta = loss_gradient_wrt_output;

        for (int i = static_cast<int>(weights_.size()) - 1; i >= 0; --i) {
            if (layer_outputs_activated_.size() <= static_cast<size_t>(i+1) || layer_activations_.size() <= static_cast<size_t>(i)) {
                 return;
            }
            Eigen::MatrixXf act_deriv_diag = activation_derivative(layer_outputs_activated_[i+1], layer_activations_[i]);
            Eigen::VectorXf delta_pre_activation = act_deriv_diag.diagonal().cwiseProduct(current_delta); 
            
            if (layer_outputs_activated_.size() <= static_cast<size_t>(i)) {
                return;
            }
            grad_weights_[i] = delta_pre_activation * layer_outputs_activated_[i].transpose();
            grad_biases_[i] = delta_pre_activation;

            if (i > 0) { current_delta = weights_[i].transpose() * delta_pre_activation; }
        }
    }

    void save_parameters_binary(const godot::Ref<godot::FileAccess>& file) const {
        file->store_32(static_cast<uint32_t>(weights_.size()));
        for (size_t i = 0; i < weights_.size(); ++i) {
            file->store_32(static_cast<uint32_t>(weights_[i].rows()));
            file->store_32(static_cast<uint32_t>(weights_[i].cols()));
            godot::PackedByteArray w_bytes = PPOC::eigen_matrix_to_byte_array(weights_[i]);
            file->store_buffer(w_bytes);
            
            file->store_32(static_cast<uint32_t>(biases_[i].size()));
            godot::PackedByteArray b_bytes = PPOC::eigen_vector_to_byte_array(biases_[i]);
            file->store_buffer(b_bytes);
        }
    }

    bool load_parameters_binary(const godot::Ref<godot::FileAccess>& file) {
        uint32_t num_layers = file->get_32(); if (file->get_error() != godot::OK) { return false;}
        if (num_layers != weights_.size()) { return false; }

        for (size_t i = 0; i < num_layers; ++i) {
            uint32_t w_r = file->get_32(); if(file->get_error()!=godot::OK)return false;
            uint32_t w_c = file->get_32(); if(file->get_error()!=godot::OK)return false;
            if (weights_[i].rows()!=static_cast<Eigen::Index>(w_r) || weights_[i].cols()!=static_cast<Eigen::Index>(w_c)) return false;
            weights_[i] = PPOC::byte_array_to_eigen_matrix(file->get_buffer(static_cast<uint64_t>(w_r)*w_c*sizeof(float)), w_r, w_c);
            if(file->get_error()!=godot::OK && !file->eof_reached()){ return false;}

            uint32_t b_s = file->get_32(); if(file->get_error()!=godot::OK)return false;
            if (biases_[i].size()!=static_cast<Eigen::Index>(b_s)) return false;
            biases_[i] = PPOC::byte_array_to_eigen_vector(file->get_buffer(static_cast<uint64_t>(b_s)*sizeof(float)), b_s);
            if(file->get_error()!=godot::OK && !file->eof_reached()){ return false;}
        }
        return true;
    }
};

class AdamOptimizerContinuous {
public:
    float lr_; float beta1_; float beta2_; float epsilon_; int t_;
    std::vector<Eigen::MatrixXf> m_weights_, v_weights_; std::vector<Eigen::VectorXf> m_biases_, v_biases_;
    AdamOptimizerContinuous(float lr=0.001f, float b1=0.9f, float b2=0.999f, float eps=1e-8f): lr_(lr),beta1_(b1),beta2_(b2),epsilon_(eps),t_(0){}
    
    void initialize(const NeuralNetworkContinuous& net) {
        m_weights_.clear(); v_weights_.clear(); m_biases_.clear(); v_biases_.clear();
        m_weights_.resize(net.weights_.size()); v_weights_.resize(net.weights_.size());
        m_biases_.resize(net.biases_.size());   v_biases_.resize(net.biases_.size());
        for (size_t i = 0; i < net.weights_.size(); ++i) {
            m_weights_[i] = Eigen::MatrixXf::Zero(net.weights_[i].rows(),net.weights_[i].cols()); 
            v_weights_[i] = Eigen::MatrixXf::Zero(net.weights_[i].rows(),net.weights_[i].cols()); 
        }
        for (size_t i = 0; i < net.biases_.size(); ++i) {
            m_biases_[i] = Eigen::VectorXf::Zero(net.biases_[i].size()); 
            v_biases_[i] = Eigen::VectorXf::Zero(net.biases_[i].size()); 
        }
        t_ = 0;
    }

    void update(NeuralNetworkContinuous& net) {
        if (net.grad_weights_.size() != net.weights_.size() || net.grad_biases_.size() != net.biases_.size()) {
            return;
        }
        t_++;
        for (size_t i=0; i<net.weights_.size(); ++i) {
            if (net.grad_weights_[i].size() == 0 || net.grad_biases_[i].size() == 0) continue;
            m_weights_[i] = beta1_*m_weights_[i] + (1.0f-beta1_)*net.grad_weights_[i];
            v_weights_[i] = beta2_*v_weights_[i] + (1.0f-beta2_)*net.grad_weights_[i].array().square().matrix();
            Eigen::MatrixXf mhw = m_weights_[i]/(1.0f-std::pow(beta1_,static_cast<float>(t_))); 
            Eigen::MatrixXf vhw = v_weights_[i]/(1.0f-std::pow(beta2_,static_cast<float>(t_)));
            net.weights_[i] -= (lr_ * mhw.array() / (vhw.array().sqrt() + epsilon_)).matrix();
            
            m_biases_[i] = beta1_*m_biases_[i] + (1.0f-beta1_)*net.grad_biases_[i];
            v_biases_[i] = beta2_*v_biases_[i] + (1.0f-beta2_)*net.grad_biases_[i].array().square().matrix();
            Eigen::VectorXf mhb = m_biases_[i]/(1.0f-std::pow(beta1_,static_cast<float>(t_))); 
            Eigen::VectorXf vhb = v_biases_[i]/(1.0f-std::pow(beta2_,static_cast<float>(t_)));
            net.biases_[i] -= (lr_ * mhb.array() / (vhb.array().sqrt() + epsilon_)).matrix();
        }
    }
    void save_state_binary(const godot::Ref<godot::FileAccess>& file) const {
        file->store_32(static_cast<uint32_t>(t_));
        for (size_t i = 0; i < m_weights_.size(); ++i) {
            file->store_buffer(PPOC::eigen_matrix_to_byte_array(m_weights_[i]));
            file->store_buffer(PPOC::eigen_matrix_to_byte_array(v_weights_[i]));
            file->store_buffer(PPOC::eigen_vector_to_byte_array(m_biases_[i]));
            file->store_buffer(PPOC::eigen_vector_to_byte_array(v_biases_[i]));
        }
    }
    bool load_state_binary(const godot::Ref<godot::FileAccess>& file, const NeuralNetworkContinuous& net_structure) {
        t_ = static_cast<int>(file->get_32()); if (file->get_error() != godot::OK) return false;
        if (m_weights_.size() != net_structure.weights_.size()) { initialize(net_structure); }

        for (size_t i = 0; i < net_structure.weights_.size(); ++i) {
            uint64_t mw_bs = static_cast<uint64_t>(net_structure.weights_[i].size()) * sizeof(float);
            m_weights_[i] = PPOC::byte_array_to_eigen_matrix(file->get_buffer(mw_bs), net_structure.weights_[i].rows(), net_structure.weights_[i].cols());
            if (file->get_error() != godot::OK && !file->eof_reached()){ return false;}
            
            uint64_t vw_bs = static_cast<uint64_t>(net_structure.weights_[i].size()) * sizeof(float);
            v_weights_[i] = PPOC::byte_array_to_eigen_matrix(file->get_buffer(vw_bs), net_structure.weights_[i].rows(), net_structure.weights_[i].cols());
            if (file->get_error() != godot::OK && !file->eof_reached()){return false;}
            
            uint64_t mb_bs = static_cast<uint64_t>(net_structure.biases_[i].size()) * sizeof(float);
            m_biases_[i] = PPOC::byte_array_to_eigen_vector(file->get_buffer(mb_bs), net_structure.biases_[i].size());
            if (file->get_error() != godot::OK && !file->eof_reached()){return false;}
            
            uint64_t vb_bs = static_cast<uint64_t>(net_structure.biases_[i].size()) * sizeof(float);
            v_biases_[i] = PPOC::byte_array_to_eigen_vector(file->get_buffer(vb_bs), net_structure.biases_[i].size());
            if (file->get_error() != godot::OK && !file->eof_reached()){return false;}
        }
        return true;
    }
};

struct TransitionContinuous {
    Eigen::VectorXf state; Eigen::VectorXf action; float reward; Eigen::VectorXf next_state;
    bool done; float old_log_prob; float old_value; float advantage; float v_target;
};

class ReplayBufferContinuous {
public:
    std::vector<TransitionContinuous> buffer_; size_t capacity_; size_t current_pos_ = 0; bool full_ = false; std::mt19937 gen;
    ReplayBufferContinuous(size_t cap, unsigned int seed = std::random_device{}()) : capacity_(cap), gen(seed){buffer_.reserve(cap);}
    void add_transition(const Eigen::VectorXf& s, const Eigen::VectorXf& a, float r, const Eigen::VectorXf& sn, bool d, float lp, float v) {
        if(buffer_.size()<capacity_){buffer_.emplace_back(TransitionContinuous{s,a,r,sn,d,lp,v,0.f,0.f});}
        else{buffer_[current_pos_]=TransitionContinuous{s,a,r,sn,d,lp,v,0.f,0.f};}
        current_pos_=(current_pos_+1)%capacity_; if(current_pos_==0 && buffer_.size()==capacity_){full_=true;}
    }
    size_t size() const { return full_ ? capacity_ : current_pos_; }
    const std::vector<TransitionContinuous>& get_all_transitions() const { return buffer_; }
    void clear() { buffer_.clear(); current_pos_ = 0; full_ = false; }
    void compute_advantages_and_returns(float gamma, float lambda_gae, float last_value_estimate) {
        if (buffer_.empty()) return; float gae_adv = 0.0f;
        for (int t = static_cast<int>(buffer_.size())-1; t>=0; --t) {
            float Vst=buffer_[t].old_value; float Rt=buffer_[t].reward; bool Dt=buffer_[t].done;
            float Vst1 = Dt ? 0.0f : ((t==static_cast<int>(buffer_.size())-1) ? last_value_estimate : buffer_[t+1].old_value);
            float delta = Rt + gamma * Vst1 - Vst;
            gae_adv = Dt ? delta : (delta + gamma*lambda_gae*gae_adv);
            buffer_[t].advantage = gae_adv; buffer_[t].v_target = gae_adv + Vst;
        }
    }
};

struct PPOCoreConfigContinuous {
    int obs_dim; int action_dim; std::vector<int> actor_hidden_dims; std::vector<int> critic_hidden_dims;
    float lr_actor; float lr_critic; float gamma; float lambda_gae; float clip_epsilon;
    int ppo_epochs; int minibatch_size; float entropy_coeff; unsigned int seed;
};

class PPOCoreContinuous {
public:
    PPOCoreConfigContinuous config_; NeuralNetworkContinuous actor_; NeuralNetworkContinuous critic_; 
    AdamOptimizerContinuous actor_optimizer_; AdamOptimizerContinuous critic_optimizer_;
    Eigen::VectorXf actor_log_stds_; Eigen::VectorXf m_log_stds_; Eigen::VectorXf v_log_stds_; int t_log_stds_ = 0;
    std::mt19937 gen_; std::normal_distribution<float> normal_dist_{0.0f, 1.0f};

    PPOCoreContinuous(const PPOCoreConfigContinuous& cfg) : 
        config_(cfg),
        actor_(cfg.obs_dim, cfg.actor_hidden_dims, cfg.action_dim, ActivationType::TANH, ActivationType::LINEAR, cfg.seed), 
        critic_(cfg.obs_dim, cfg.critic_hidden_dims, 1, ActivationType::TANH, ActivationType::LINEAR, cfg.seed + 1),
        actor_optimizer_(cfg.lr_actor), critic_optimizer_(cfg.lr_critic), gen_(cfg.seed + 2) {
        actor_optimizer_.initialize(actor_); critic_optimizer_.initialize(critic_);
        actor_log_stds_ = Eigen::VectorXf::Zero(config_.action_dim); 
        m_log_stds_ = Eigen::VectorXf::Zero(config_.action_dim); v_log_stds_ = Eigen::VectorXf::Zero(config_.action_dim); t_log_stds_ = 0;
    }

    std::tuple<Eigen::VectorXf, float, float> select_action_details(const Eigen::VectorXf& observation) {
        Eigen::VectorXf mus = actor_.forward(observation);
        Eigen::VectorXf stds = actor_log_stds_.array().exp().matrix();
        Eigen::VectorXf actions(config_.action_dim);
        for (int i=0; i<config_.action_dim; ++i) { actions[i] = mus[i] + stds[i] * normal_dist_(gen_); }
        float log_prob = 0.0f;
        for (int i=0; i<config_.action_dim; ++i) {
            float t1=((actions[i]-mus[i])/(stds[i]+1e-8f)); t1*=t1; float t2=2.0f*actor_log_stds_[i]; float t3=std::log(2.0f*static_cast<float>(M_PI));
            log_prob -= 0.5f*(t1+t2+t3);
        }
        return {actions, log_prob, critic_.forward(observation)[0]};
    }

    void update(ReplayBufferContinuous& buffer, float last_value_estimate_for_gae) {
        buffer.compute_advantages_and_returns(config_.gamma, config_.lambda_gae, last_value_estimate_for_gae);
        const auto& transitions = buffer.get_all_transitions(); if (transitions.empty()) return;
        std::vector<float> adv_norm(transitions.size()); float adv_m=0.0f;
        for(size_t k=0;k<transitions.size();++k){adv_norm[k]=transitions[k].advantage; adv_m+=adv_norm[k];}
        if(!transitions.empty()){adv_m/=static_cast<float>(transitions.size());} float adv_sqs=0.0f;
        for(size_t k=0;k<transitions.size();++k){adv_norm[k]-=adv_m; adv_sqs+=adv_norm[k]*adv_norm[k];}
        float adv_std=(transitions.size()>1)?std::sqrt(adv_sqs/static_cast<float>(transitions.size()-1)):1.0f;
        if(adv_std<1e-8f)adv_std=1e-8f; for(size_t k=0;k<transitions.size();++k){adv_norm[k]/=adv_std;}
        std::vector<int> idx(transitions.size()); std::iota(idx.begin(),idx.end(),0);

        for (int epoch=0; epoch<config_.ppo_epochs; ++epoch) {
            std::shuffle(idx.begin(),idx.end(),gen_);
            for (size_t i=0; i<transitions.size(); i+=config_.minibatch_size) {
                size_t batch_e = std::min(i+config_.minibatch_size, transitions.size()); if(batch_e<=i)continue;
                std::vector<Eigen::MatrixXf> b_actor_gw(actor_.weights_.size()); std::vector<Eigen::VectorXf> b_actor_gb(actor_.biases_.size());
                for(size_t k=0;k<actor_.weights_.size();++k){b_actor_gw[k].setZero(actor_.weights_[k].rows(),actor_.weights_[k].cols()); b_actor_gb[k].setZero(actor_.biases_[k].size());}
                std::vector<Eigen::MatrixXf> b_critic_gw(critic_.weights_.size()); std::vector<Eigen::VectorXf> b_critic_gb(critic_.biases_.size());
                for(size_t k=0;k<critic_.weights_.size();++k){b_critic_gw[k].setZero(critic_.weights_[k].rows(),critic_.weights_[k].cols()); b_critic_gb[k].setZero(critic_.biases_[k].size());}
                Eigen::VectorXf b_actor_lstd_g=Eigen::VectorXf::Zero(config_.action_dim);

                for (size_t ji=i; ji<batch_e; ++ji) {
                    int ti = idx[ji]; const auto& tr=transitions[ti]; float c_adv=adv_norm[ti];
                    Eigen::VectorXf c_mus=actor_.forward(tr.state); Eigen::VectorXf c_stds=actor_log_stds_.array().exp().matrix();
                    float n_lp=0.0f;
                    for(int k=0;k<config_.action_dim;++k){float t1=((tr.action[k]-c_mus[k])/(c_stds[k]+1e-8f));t1*=t1; float t2=2.0f*actor_log_stds_[k]; float t3=std::log(2.0f*static_cast<float>(M_PI)); n_lp-=0.5f*(t1+t2+t3);}
                    float rto=std::exp(n_lp-tr.old_log_prob); float s1=rto*c_adv; float s2=std::clamp(rto,1.0f-config_.clip_epsilon,1.0f+config_.clip_epsilon)*c_adv;
                    float ent=0.0f; for(int k=0;k<config_.action_dim;++k){ent+=0.5f*(std::log(2.0f*static_cast<float>(M_PI))+1.0f+2.0f*actor_log_stds_[k]);}
                    Eigen::VectorXf glp_mus=Eigen::VectorXf::Zero(config_.action_dim); Eigen::VectorXf glp_lstds=Eigen::VectorXf::Zero(config_.action_dim);
                    for(int k=0;k<config_.action_dim;++k){float ak=tr.action[k];float muk=c_mus[k];float sk2=c_stds[k]*c_stds[k]+1e-8f;float sk=c_stds[k]+1e-8f;glp_mus[k]=(ak-muk)/sk2;float tslg=(ak-muk)/sk;glp_lstds[k]=tslg*tslg-1.0f;}
                    
                    float policy_loss_grad_multiplier;
                    if(s1<=s2){policy_loss_grad_multiplier = -c_adv*rto;} 
                    else {
                        if((c_adv>0&&rto>=(1.0f+config_.clip_epsilon))||(c_adv<0&&rto<=(1.0f-config_.clip_epsilon))){
                            policy_loss_grad_multiplier=0.0f;
                        } else {
                            policy_loss_grad_multiplier=-c_adv*std::clamp(rto,1.0f-config_.clip_epsilon,1.0f+config_.clip_epsilon);
                        }
                    }
                    Eigen::VectorXf pobj_g_mus=policy_loss_grad_multiplier*glp_mus; Eigen::VectorXf pobj_g_lstds=policy_loss_grad_multiplier*glp_lstds;
                    Eigen::VectorXf ent_g_lstds=Eigen::VectorXf::Ones(config_.action_dim);
                    Eigen::VectorXf tot_a_g_mus=pobj_g_mus; Eigen::VectorXf tot_a_g_lstds=pobj_g_lstds-config_.entropy_coeff*ent_g_lstds;
                    actor_.backward(tot_a_g_mus); for(size_t k=0;k<actor_.weights_.size();++k){b_actor_gw[k]+=actor_.grad_weights_[k];b_actor_gb[k]+=actor_.grad_biases_[k];}
                    b_actor_lstd_g+=tot_a_g_lstds;
                    Eigen::VectorXf c_val_out=critic_.forward(tr.state);float c_val=c_val_out[0];Eigen::VectorXf cri_g_out=Eigen::VectorXf::Constant(1,2.0f*(c_val-tr.v_target));
                    critic_.backward(cri_g_out); for(size_t k=0;k<critic_.weights_.size();++k){b_critic_gw[k]+=critic_.grad_weights_[k];b_critic_gb[k]+=critic_.grad_biases_[k];}
                }
                float mbsf=static_cast<float>(batch_e-i); if(mbsf>0){
                    for(size_t k=0;k<actor_.weights_.size();++k){actor_.grad_weights_[k]=b_actor_gw[k]/mbsf;actor_.grad_biases_[k]=b_actor_gb[k]/mbsf;} actor_optimizer_.update(actor_);
                    Eigen::VectorXf avg_lstd_g=b_actor_lstd_g/mbsf; t_log_stds_++;
                    float b1=actor_optimizer_.beta1_;float b2=actor_optimizer_.beta2_;float eps=actor_optimizer_.epsilon_;
                    m_log_stds_=b1*m_log_stds_+(1.0f-b1)*avg_lstd_g; v_log_stds_=b2*v_log_stds_+(1.0f-b2)*avg_lstd_g.array().square().matrix();
                    Eigen::VectorXf mh=m_log_stds_/(1.0f-std::pow(b1,static_cast<float>(t_log_stds_))); Eigen::VectorXf vh=v_log_stds_/(1.0f-std::pow(b2,static_cast<float>(t_log_stds_)));
                    actor_log_stds_-=(config_.lr_actor*mh.array()/(vh.array().sqrt()+eps)).matrix();
                    for(size_t k=0;k<critic_.weights_.size();++k){critic_.grad_weights_[k]=b_critic_gw[k]/mbsf;critic_.grad_biases_[k]=b_critic_gb[k]/mbsf;} critic_optimizer_.update(critic_);
                }
            }
        }
        buffer.clear();
    }

    void save_model_binary(const godot::Ref<godot::FileAccess>& file) const {
        actor_.save_parameters_binary(file); critic_.save_parameters_binary(file);
        actor_optimizer_.save_state_binary(file); critic_optimizer_.save_state_binary(file);
        file->store_32(static_cast<uint32_t>(actor_log_stds_.size()));
        file->store_buffer(PPOC::eigen_vector_to_byte_array(actor_log_stds_));
        file->store_32(static_cast<uint32_t>(t_log_stds_));
        file->store_buffer(PPOC::eigen_vector_to_byte_array(m_log_stds_));
        file->store_buffer(PPOC::eigen_vector_to_byte_array(v_log_stds_));
    }

    bool load_model_binary(const godot::Ref<godot::FileAccess>& file) {
        if (!actor_.load_parameters_binary(file)) { return false;}
        if (!critic_.load_parameters_binary(file)) {return false;}
        if (!actor_optimizer_.load_state_binary(file, actor_)) {return false;}
        if (!critic_optimizer_.load_state_binary(file, critic_)) {return false;}
        
        uint32_t lstds_s = file->get_32(); if(file->get_error()!=godot::OK){return false;}
        if (lstds_s != static_cast<uint32_t>(config_.action_dim)) { return false;}
        actor_log_stds_ = PPOC::byte_array_to_eigen_vector(file->get_buffer(lstds_s * sizeof(float)), lstds_s); 
        if(file->get_error()!=godot::OK && !file->eof_reached()){return false;}
        
        t_log_stds_ = static_cast<int>(file->get_32()); if(file->get_error()!=godot::OK){return false;}
        m_log_stds_.resize(lstds_s); v_log_stds_.resize(lstds_s); 
        m_log_stds_ = PPOC::byte_array_to_eigen_vector(file->get_buffer(lstds_s * sizeof(float)), lstds_s); 
        if(file->get_error()!=godot::OK && !file->eof_reached()){return false;}
        v_log_stds_ = PPOC::byte_array_to_eigen_vector(file->get_buffer(lstds_s * sizeof(float)), lstds_s); 
        if(file->get_error()!=godot::OK && !file->eof_reached()){return false;}
        return true;
    }
};

} // Ende PPOInternalContinuous Namespace


PPOC::PPOC() {}
PPOC::~PPOC() {}

void PPOC::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("initialize", "observation_dim", "action_dim", "actor_hidden_dims", "critic_hidden_dims", "lr_actor", "lr_critic", "gamma", "lambda_gae", "clip_epsilon", "ppo_epochs", "minibatch_size", "entropy_coeff", "buffer_size", "seed"), &PPOC::initialize, DEFVAL(0.0003f), DEFVAL(0.001f), DEFVAL(0.99f), DEFVAL(0.95f), DEFVAL(0.2f), DEFVAL(10), DEFVAL(64), DEFVAL(0.01f), DEFVAL(2048), DEFVAL(-1));
    godot::ClassDB::bind_method(godot::D_METHOD("get_action", "observation_array"), &PPOC::get_action);
    godot::ClassDB::bind_method(godot::D_METHOD("store_experience", "reward", "next_observation_array", "done"), &PPOC::store_experience);
    godot::ClassDB::bind_method(godot::D_METHOD("train"), &PPOC::train);
    godot::ClassDB::bind_method(godot::D_METHOD("save_model", "file_path"), &PPOC::save_model);
    godot::ClassDB::bind_method(godot::D_METHOD("load_model", "file_path"), &PPOC::load_model);
}

void PPOC::initialize(int p_observation_dim, int p_action_dim,
                    const godot::Array& p_actor_hidden_dims, const godot::Array& p_critic_hidden_dims,
                    float p_lr_actor, float p_lr_critic,
                    float p_gamma, float p_lambda_gae, float p_clip_epsilon,
                    int p_ppo_epochs, int p_minibatch_size, float p_entropy_coeff,
                    int p_buffer_size, int p_seed) {
    
    PPOInternalContinuous::PPOCoreConfigContinuous ppo_config;
    observation_dim_continuous_ = p_observation_dim;
    action_dim_continuous_ = p_action_dim;

    ppo_config.obs_dim = observation_dim_continuous_;
    ppo_config.action_dim = action_dim_continuous_;

    for(int i = 0; i < p_actor_hidden_dims.size(); ++i) {
        ppo_config.actor_hidden_dims.push_back(p_actor_hidden_dims[i]);
    }
    for(int i = 0; i < p_critic_hidden_dims.size(); ++i) {
        ppo_config.critic_hidden_dims.push_back(p_critic_hidden_dims[i]);
    }

    ppo_config.lr_actor = p_lr_actor;
    ppo_config.lr_critic = p_lr_critic;
    ppo_config.gamma = p_gamma;
    ppo_config.lambda_gae = p_lambda_gae;
    ppo_config.clip_epsilon = p_clip_epsilon;
    ppo_config.ppo_epochs = p_ppo_epochs;
    ppo_config.minibatch_size = p_minibatch_size;
    ppo_config.entropy_coeff = p_entropy_coeff;

    if (p_seed < 0) {
        ppo_config.seed = std::random_device{}();
    } else {
        ppo_config.seed = static_cast<unsigned int>(p_seed);
    }
    
    buffer_size_continuous_ = p_buffer_size;
    train_every_n_steps_continuous_ = buffer_size_continuous_;

    if (observation_dim_continuous_ <= 0 || action_dim_continuous_ <= 0 || ppo_config.actor_hidden_dims.empty() || ppo_config.critic_hidden_dims.empty()) {
        godot::UtilityFunctions::print("PPOC initialization failed: Invalid dimensions or empty hidden layers.");
        initialized_continuous_ = false;
        return;
    }
    
    ppo_core_continuous_ = std::make_unique<PPOInternalContinuous::PPOCoreContinuous>(ppo_config);
    replay_buffer_continuous_ = std::make_unique<PPOInternalContinuous::ReplayBufferContinuous>(buffer_size_continuous_, ppo_config.seed + 3);
    initialized_continuous_ = true;
    training_counter_continuous_ = 0;
    current_observation_continuous_.resize(0);
    current_action_vector_continuous_.resize(0);
}


godot::PackedFloat32Array PPOC::get_action(const godot::PackedFloat32Array& observation_array) {
    if (!initialized_continuous_) { return godot::PackedFloat32Array(); }
    if (observation_array.size()!=observation_dim_continuous_) { return godot::PackedFloat32Array(); }
    current_observation_continuous_ = packed_array_to_eigen_vector(observation_array);
    auto [act_vec, lp, val_est] = ppo_core_continuous_->select_action_details(current_observation_continuous_);
    current_action_vector_continuous_ = act_vec; current_action_log_prob_continuous_ = lp; current_value_estimate_continuous_ = val_est;
    return eigen_vector_to_packed_array(current_action_vector_continuous_);
}

void PPOC::store_experience(float reward, const godot::PackedFloat32Array& next_observation_array, bool done) {
    if (!initialized_continuous_) { return; }
    if (next_observation_array.size()!=observation_dim_continuous_) { return; }
    if (current_observation_continuous_.size()!=observation_dim_continuous_ || current_action_vector_continuous_.size()!=action_dim_continuous_) { 
        return;
    }
    Eigen::VectorXf next_obs = packed_array_to_eigen_vector(next_observation_array);
    replay_buffer_continuous_->add_transition(current_observation_continuous_, current_action_vector_continuous_, reward, next_obs, done, current_action_log_prob_continuous_, current_value_estimate_continuous_);
    training_counter_continuous_++; current_observation_continuous_.resize(0); current_action_vector_continuous_.resize(0);
    if (training_counter_continuous_ >= train_every_n_steps_continuous_ ) { train(); training_counter_continuous_ = 0; }
}

void PPOC::train() {
    if (!initialized_continuous_) {return; }
    if (replay_buffer_continuous_->size() == 0) { return; }
    float last_val_gae = 0.0f;
    if (!replay_buffer_continuous_->buffer_.empty()) { 
        const auto& last_trans = replay_buffer_continuous_->buffer_.back();
        if (!last_trans.done) { last_val_gae = ppo_core_continuous_->critic_.forward(last_trans.next_state)[0]; }
    }
    ppo_core_continuous_->update(*replay_buffer_continuous_, last_val_gae);
}

bool PPOC::save_model(const godot::String& file_path) {
    if (!initialized_continuous_ || !ppo_core_continuous_) { return false; }
    godot::Error err; godot::Ref<godot::FileAccess> file = godot::FileAccess::open(file_path, godot::FileAccess::WRITE);
    if (file.is_null() || !file->is_open()) { err=godot::FileAccess::get_open_error(); return false; }
    file->store_32(PPO_AGENT_CONTINUOUS_MAGIC_NUMBER); file->store_32(PPO_AGENT_CONTINUOUS_FORMAT_VERSION);
    file->store_32(static_cast<uint32_t>(observation_dim_continuous_)); file->store_32(static_cast<uint32_t>(action_dim_continuous_));
    ppo_core_continuous_->save_model_binary(file);
    if (file->get_error() != godot::OK) {  file->close(); return false; }
    file->close(); return true;
}

bool PPOC::load_model(const godot::String& file_path) {
    if (!initialized_continuous_ || !ppo_core_continuous_) { return false; }
    if (!godot::FileAccess::file_exists(file_path)) {  return false; }
    godot::Error err; godot::Ref<godot::FileAccess> file = godot::FileAccess::open(file_path, godot::FileAccess::READ);
    if (file.is_null() || !file->is_open()) { err=godot::FileAccess::get_open_error();  return false; }
    uint32_t magic = file->get_32(); if(file->get_error()!=godot::OK || magic!=PPO_AGENT_CONTINUOUS_MAGIC_NUMBER){file->close();return false;}
    uint32_t version = file->get_32(); if(file->get_error()!=godot::OK || version>PPO_AGENT_CONTINUOUS_FORMAT_VERSION){file->close();return false;}
    uint32_t s_obs_d = file->get_32(); uint32_t s_act_d = file->get_32(); if(file->get_error()!=godot::OK) {file->close(); return false;}
    if (static_cast<int>(s_obs_d)!=observation_dim_continuous_ || static_cast<int>(s_act_d)!=action_dim_continuous_) {
         file->close(); return false; 
    }
    if (!ppo_core_continuous_->load_model_binary(file)) {file->close(); return false; }
    if (file->get_error()!=godot::OK && !file->eof_reached()) { file->close(); return false; }
    file->close(); return true;
}
