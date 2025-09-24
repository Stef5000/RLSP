#include "ppo.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/file_access.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <algorithm>
#include <random>

Eigen::VectorXf PPO::packed_array_to_eigen(const godot::PackedFloat32Array& p_array) {
    Eigen::VectorXf vec(p_array.size());
    for (int i = 0; i < p_array.size(); ++i) {
        vec[i] = p_array[i];
    }
    return vec;
}

godot::PackedFloat32Array PPO::eigen_to_packed_array(const Eigen::VectorXf& p_vector) {
    godot::PackedFloat32Array arr;
    arr.resize(p_vector.size());
    for (int i = 0; i < p_vector.size(); ++i) {
        arr[i] = p_vector[i];
    }
    return arr;
}

namespace PPOInternal {

bool write_eigen_matrix(godot::FileAccess* file, const Eigen::MatrixXf& matrix) {
    if (!file) return false;
    file->store_64(static_cast<uint64_t>(matrix.rows()));
    file->store_64(static_cast<uint64_t>(matrix.cols()));
    if (matrix.size() > 0) {
        file->store_buffer(reinterpret_cast<const uint8_t*>(matrix.data()), matrix.size() * sizeof(float));
    }
    return file->get_error() == godot::OK;
}

bool read_eigen_matrix(godot::FileAccess* file, Eigen::MatrixXf& matrix) {
    if (!file) return false;
    uint64_t rows_u = file->get_64();
    uint64_t cols_u = file->get_64();
    Eigen::Index rows = static_cast<Eigen::Index>(rows_u);
    Eigen::Index cols = static_cast<Eigen::Index>(cols_u);

    if (rows < 0 || cols < 0) {
        return false;
    }
    
    matrix.resize(rows, cols);
    if (matrix.size() > 0) {
        godot::PackedByteArray buffer;
        buffer.resize(matrix.size() * sizeof(float));
        uint64_t bytes_read = file->get_buffer(buffer.ptrw(), buffer.size());
        if (bytes_read != static_cast<uint64_t>(buffer.size())) {
            return false;
        }
        memcpy(matrix.data(), buffer.ptr(), buffer.size());
    }
    return file->get_error() == godot::OK;
}

bool write_eigen_vector(godot::FileAccess* file, const Eigen::VectorXf& vector) {
    if (!file) return false;
    file->store_64(static_cast<uint64_t>(vector.size()));
    if (vector.size() > 0) {
        file->store_buffer(reinterpret_cast<const uint8_t*>(vector.data()), vector.size() * sizeof(float));
    }
    return file->get_error() == godot::OK;
}

bool read_eigen_vector(godot::FileAccess* file, Eigen::VectorXf& vector) {
    if (!file) return false;
    uint64_t size_u = file->get_64();
    Eigen::Index size = static_cast<Eigen::Index>(size_u);
     if (size < 0 ) { 
        return false;
    }
    vector.resize(size);
    if (vector.size() > 0) {
        godot::PackedByteArray buffer;
        buffer.resize(vector.size() * sizeof(float));
        uint64_t bytes_read = file->get_buffer(buffer.ptrw(), buffer.size());
        if (bytes_read != static_cast<uint64_t>(buffer.size())) {
            return false;
        }
        memcpy(vector.data(), buffer.ptr(), buffer.size());
    }
    return file->get_error() == godot::OK;
}

// --- Activation Functions ---
enum class ActivationType { RELU, TANH, LINEAR, SOFTMAX };
Eigen::VectorXf apply_activation(const Eigen::VectorXf& x, ActivationType type) {
    switch (type) {
        case ActivationType::RELU: return x.unaryExpr([](float v){ return std::max(0.0f, v); });
        case ActivationType::TANH: return x.unaryExpr([](float v){ return std::tanh(v); });
        case ActivationType::LINEAR: return x;
        case ActivationType::SOFTMAX: {
            if (x.size() == 0) return x;
            Eigen::VectorXf exp_x = (x.array() - x.maxCoeff()).exp();
            float sum_exp = exp_x.sum();
            if (std::abs(sum_exp) < 1e-8) return Eigen::VectorXf::Constant(x.size(), 1.0f/static_cast<float>(x.size()));
            return exp_x / sum_exp;
        }
        default: return x; 
    }
}
Eigen::MatrixXf activation_derivative(const Eigen::VectorXf& activated_output, ActivationType type) {
    Eigen::VectorXf derivatives;
    switch (type) {
        case ActivationType::RELU: derivatives = activated_output.unaryExpr([](float v){ return v > 0.0f ? 1.0f : 0.0f; }); break;
        case ActivationType::TANH: derivatives = activated_output.unaryExpr([](float v){ return 1.0f - v * v; }); break;
        case ActivationType::LINEAR: derivatives = Eigen::VectorXf::Ones(activated_output.size()); break;
        case ActivationType::SOFTMAX: 
            derivatives = Eigen::VectorXf::Ones(activated_output.size()); break;
        default: derivatives = Eigen::VectorXf::Ones(activated_output.size());
    }
    return derivatives.asDiagonal();
}

// --- Neural Network ---
class NeuralNetwork {
public:
    std::vector<Eigen::MatrixXf> weights_; std::vector<Eigen::VectorXf> biases_;
    std::vector<ActivationType> layer_activations_; std::vector<Eigen::VectorXf> layer_outputs_pre_activation_; 
    std::vector<Eigen::VectorXf> layer_outputs_activated_; std::mt19937 gen; 
    std::vector<Eigen::MatrixXf> grad_weights_; std::vector<Eigen::VectorXf> grad_biases_;

    NeuralNetwork(int i_dim, const std::vector<int>& h_dims, int o_dim, ActivationType h_act, ActivationType o_act, unsigned int sd = std::random_device{}()) : gen(sd) {
        int c_dim = i_dim;
        for (size_t i = 0; i < h_dims.size(); ++i) { add_layer(c_dim, h_dims[i], h_act); c_dim = h_dims[i]; }
        add_layer(c_dim, o_dim, o_act);
    }
    void add_layer(int i_dim, int o_dim, ActivationType act) {
        float limit = std::sqrt(6.0f / static_cast<float>(i_dim + o_dim)); std::uniform_real_distribution<float> dist(-limit, limit);
        Eigen::MatrixXf W(o_dim, i_dim); for(int r=0;r<W.rows();++r)for(int c=0;c<W.cols();++c)W(r,c)=dist(gen); weights_.push_back(W);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(o_dim); biases_.push_back(b); layer_activations_.push_back(act);
    }
    Eigen::VectorXf forward(const Eigen::VectorXf& in) {
        layer_outputs_pre_activation_.clear(); layer_outputs_activated_.clear(); Eigen::VectorXf c_o = in;
        layer_outputs_activated_.push_back(c_o); 
        for (size_t i=0;i<weights_.size();++i){Eigen::VectorXf pre_a=weights_[i]*c_o+biases_[i];layer_outputs_pre_activation_.push_back(pre_a);c_o=apply_activation(pre_a,layer_activations_[i]);layer_outputs_activated_.push_back(c_o);}
        return c_o;
    }
    void backward(const Eigen::VectorXf& loss_grad) {
        grad_weights_.assign(weights_.size(),Eigen::MatrixXf()); grad_biases_.assign(biases_.size(),Eigen::VectorXf());
        Eigen::VectorXf c_d=loss_grad;
        for (int i=weights_.size()-1;i>=0;--i){
            Eigen::MatrixXf act_deriv_d=activation_derivative(layer_outputs_activated_[i+1],layer_activations_[i]);
            Eigen::VectorXf d_pre_a=act_deriv_d.diagonal().cwiseProduct(c_d);
            grad_weights_[i]=d_pre_a*layer_outputs_activated_[i].transpose(); grad_biases_[i]=d_pre_a;
            if(i>0){c_d=weights_[i].transpose()*d_pre_a;}
        }
    }
    bool save_parameters(godot::FileAccess* file) const {
        if(!file)return false; file->store_32(static_cast<uint32_t>(weights_.size()));
        for(const auto& W:weights_)if(!write_eigen_matrix(file,W))return false;
        file->store_32(static_cast<uint32_t>(biases_.size()));
        for(const auto& b:biases_)if(!write_eigen_vector(file,b))return false;
        return file->get_error()==godot::OK;
    }
    bool load_parameters(godot::FileAccess* file) {
        if(!file)return false; uint32_t num_w_l=file->get_32();
        if(num_w_l!=weights_.size()){ return false;}
        for(size_t i=0;i<num_w_l;++i)if(!read_eigen_matrix(file,weights_[i]))return false;
        uint32_t num_b_l=file->get_32();
        if(num_b_l!=biases_.size()){ return false;}
        for(size_t i=0;i<num_b_l;++i)if(!read_eigen_vector(file,biases_[i]))return false;
        return file->get_error()==godot::OK;
    }
};

// --- Adam Optimizer ---
class AdamOptimizer {
public:
    float lr_; float beta1_, beta2_, epsilon_; int t_; 
    std::vector<Eigen::MatrixXf> m_weights_, v_weights_; std::vector<Eigen::VectorXf> m_biases_, v_biases_;

    AdamOptimizer(float lr=0.001f, float b1=0.9f, float b2=0.999f, float eps=1e-8f):lr_(lr),beta1_(b1),beta2_(b2),epsilon_(eps),t_(0){}
    void initialize(const NeuralNetwork& net) {
        m_weights_.clear();v_weights_.clear();m_biases_.clear();v_biases_.clear();
        for(const auto&W:net.weights_){m_weights_.push_back(Eigen::MatrixXf::Zero(W.rows(),W.cols()));v_weights_.push_back(Eigen::MatrixXf::Zero(W.rows(),W.cols()));}
        for(const auto&b:net.biases_){m_biases_.push_back(Eigen::VectorXf::Zero(b.size()));v_biases_.push_back(Eigen::VectorXf::Zero(b.size()));} t_=0;
    }
    void update(NeuralNetwork& net) {
        t_++; for(size_t i=0;i<net.weights_.size();++i){
            m_weights_[i]=beta1_*m_weights_[i]+(1-beta1_)*net.grad_weights_[i];
            v_weights_[i]=beta2_*v_weights_[i]+(1-beta2_)*net.grad_weights_[i].array().square().matrix();
            Eigen::MatrixXf mhw=m_weights_[i]/(1-std::pow(beta1_,t_)); Eigen::MatrixXf vhw=v_weights_[i]/(1-std::pow(beta2_,t_));
            net.weights_[i]-=(lr_*mhw.array()/(vhw.array().sqrt()+epsilon_)).matrix();
            m_biases_[i]=beta1_*m_biases_[i]+(1-beta1_)*net.grad_biases_[i];
            v_biases_[i]=beta2_*v_biases_[i]+(1-beta2_)*net.grad_biases_[i].array().square().matrix();
            Eigen::VectorXf mhb=m_biases_[i]/(1-std::pow(beta1_,t_)); Eigen::VectorXf vhb=v_biases_[i]/(1-std::pow(beta2_,t_));
            net.biases_[i]-=(lr_*mhb.array()/(vhb.array().sqrt()+epsilon_)).matrix();
        }
    }
    bool save_state(godot::FileAccess* file) const {
        if(!file)return false; file->store_32(static_cast<uint32_t>(t_));
        for(const auto&mW:m_weights_)if(!write_eigen_matrix(file,mW))return false;
        for(const auto&vW:v_weights_)if(!write_eigen_matrix(file,vW))return false;
        for(const auto&mb:m_biases_)if(!write_eigen_vector(file,mb))return false;
        for(const auto&vb:v_biases_)if(!write_eigen_vector(file,vb))return false;
        return file->get_error()==godot::OK;
    }
    bool load_state(godot::FileAccess* file, const NeuralNetwork& net_s) {
        if(!file)return false; t_=static_cast<int>(file->get_32());
        if(m_weights_.size()!=net_s.weights_.size()){ initialize(net_s); }
        for(size_t i=0;i<m_weights_.size();++i)if(!read_eigen_matrix(file,m_weights_[i]))return false;
        for(size_t i=0;i<v_weights_.size();++i)if(!read_eigen_matrix(file,v_weights_[i]))return false;
        for(size_t i=0;i<m_biases_.size();++i)if(!read_eigen_vector(file,m_biases_[i]))return false;
        for(size_t i=0;i<v_biases_.size();++i)if(!read_eigen_vector(file,v_biases_[i]))return false;
        return file->get_error()==godot::OK;
    }
};

// --- Replay Buffer ---
struct Transition { Eigen::VectorXf state; int action; float reward; Eigen::VectorXf next_state; bool done; float old_log_prob; float old_value; float advantage; float v_target; };
class ReplayBuffer {
public:
    std::vector<Transition> buffer_; size_t capacity_; size_t current_pos_=0; bool full_=false; std::mt19937 gen;
    ReplayBuffer(size_t cap,unsigned int s=std::random_device{}()):capacity_(cap),gen(s){buffer_.reserve(cap);}
    void add_transition(const Eigen::VectorXf& s,int a,float r,const Eigen::VectorXf& sn,bool d,float lp,float v){ if(buffer_.size()<capacity_){buffer_.emplace_back(Transition{s,a,r,sn,d,lp,v,0.f,0.f});} else{buffer_[current_pos_]=Transition{s,a,r,sn,d,lp,v,0.f,0.f};} current_pos_=(current_pos_+1)%capacity_;if(current_pos_==0&&buffer_.size()==capacity_)full_=true; }
    size_t size() const { return full_?capacity_:current_pos_; }
    void compute_advantages_and_returns(float gamma,float lambda_gae,float last_v){ if(buffer_.empty())return;float gae_adv=0.0f; for(int t=buffer_.size()-1;t>=0;--t){ float Vst=buffer_[t].old_value;float Rt=buffer_[t].reward;bool Dt=buffer_[t].done; float Vst1=Dt?0.0f:((t==static_cast<int>(buffer_.size())-1)?last_v:buffer_[t+1].old_value); float delta=Rt+gamma*Vst1-Vst;gae_adv=Dt?delta:(delta+gamma*lambda_gae*gae_adv); buffer_[t].advantage=gae_adv;buffer_[t].v_target=gae_adv+Vst; } }
    const std::vector<Transition>& get_all_transitions() const { return buffer_; }
    void clear() { buffer_.clear();current_pos_=0;full_=false; }
};

// --- PPO Core Logic ---
struct PPOCoreConfig { int obs_dim; int action_dim; std::vector<int> actor_hidden_dims; std::vector<int> critic_hidden_dims; float lr_actor; float lr_critic; float gamma; float lambda_gae; float clip_epsilon; int ppo_epochs; int minibatch_size; float entropy_coeff; unsigned int seed; };
class PPOCore {
public:
    PPOCoreConfig config_; NeuralNetwork actor_; NeuralNetwork critic_;
    AdamOptimizer actor_optimizer_; AdamOptimizer critic_optimizer_; std::mt19937 gen_; 

    PPOCore(const PPOCoreConfig& cfg):config_(cfg),
        actor_(cfg.obs_dim,cfg.actor_hidden_dims,cfg.action_dim,ActivationType::TANH,ActivationType::LINEAR,cfg.seed), 
        critic_(cfg.obs_dim,cfg.critic_hidden_dims,1,ActivationType::TANH,ActivationType::LINEAR,cfg.seed+1),
        actor_optimizer_(cfg.lr_actor),critic_optimizer_(cfg.lr_critic),gen_(cfg.seed+2) {
        actor_optimizer_.initialize(actor_); critic_optimizer_.initialize(critic_);
    }
    std::tuple<int,float,float> select_action_details(const Eigen::VectorXf& obs){
        Eigen::VectorXf logits=actor_.forward(obs);Eigen::VectorXf probs=apply_activation(logits,ActivationType::SOFTMAX);
        std::discrete_distribution<int>dist(probs.data(),probs.data()+probs.size());int act=dist(gen_);
        float lp=std::log(probs[act]+1e-8f);float val=critic_.forward(obs)[0];return{act,lp,val};
    }
    void update(ReplayBuffer& buf,float last_val_gae){
        buf.compute_advantages_and_returns(config_.gamma,config_.lambda_gae,last_val_gae);
        const auto& trans=buf.get_all_transitions();if(trans.empty())return;
        std::vector<float> adv_n(trans.size());float adv_m=0.0f;
        for(size_t k=0;k<trans.size();++k){adv_n[k]=trans[k].advantage;adv_m+=adv_n[k];}
        if(!trans.empty())adv_m/=trans.size();float adv_sqs=0.0f;
        for(size_t k=0;k<trans.size();++k){adv_n[k]-=adv_m;adv_sqs+=adv_n[k]*adv_n[k];}
        float adv_std=(trans.size()>1)?std::sqrt(adv_sqs/(trans.size()-1)):1.0f;
        if(adv_std<1e-8f)adv_std=1e-8f;for(size_t k=0;k<trans.size();++k)adv_n[k]/=adv_std;
        std::vector<int>idx(trans.size());std::iota(idx.begin(),idx.end(),0);
        for(int ep=0;ep<config_.ppo_epochs;++ep){
            std::shuffle(idx.begin(),idx.end(),gen_);
            for(size_t i=0;i<trans.size();i+=config_.minibatch_size){
                size_t b_end=std::min(i+config_.minibatch_size,trans.size());if(b_end<=i)continue;
                std::vector<Eigen::MatrixXf>b_agw(actor_.weights_.size());std::vector<Eigen::VectorXf>b_agb(actor_.biases_.size());
                for(size_t k=0;k<actor_.weights_.size();++k){b_agw[k].setZero(actor_.weights_[k].rows(),actor_.weights_[k].cols());b_agb[k].setZero(actor_.biases_[k].size());}
                std::vector<Eigen::MatrixXf>b_cgw(critic_.weights_.size());std::vector<Eigen::VectorXf>b_cgb(critic_.biases_.size());
                for(size_t k=0;k<critic_.weights_.size();++k){b_cgw[k].setZero(critic_.weights_[k].rows(),critic_.weights_[k].cols());b_cgb[k].setZero(critic_.biases_[k].size());}
                for(size_t ji=i;ji<b_end;++ji){
                    const auto& tr=trans[idx[ji]];float c_adv=adv_n[idx[ji]];
                    Eigen::VectorXf c_logits=actor_.forward(tr.state);Eigen::VectorXf c_probs=apply_activation(c_logits,ActivationType::SOFTMAX);
                    float c_lp=std::log(c_probs[tr.action]+1e-8f);float rto=std::exp(c_lp-tr.old_log_prob);
                    float s1=rto*c_adv;float s2=std::clamp(rto,1.0f-config_.clip_epsilon,1.0f+config_.clip_epsilon)*c_adv;
                    Eigen::VectorXf glp_logits=Eigen::VectorXf::Zero(config_.action_dim);glp_logits[tr.action]=1.0f;glp_logits-=c_probs;
                    float pgf;if(s1<s2){pgf=-c_adv*rto;}else{if((c_adv>0&&rto>=(1.0f+config_.clip_epsilon))||(c_adv<0&&rto<=(1.0f-config_.clip_epsilon)))pgf=0.0f;else pgf=-c_adv*std::clamp(rto,1.0f-config_.clip_epsilon,1.0f+config_.clip_epsilon);}
                    Eigen::VectorXf plg_logits=pgf*glp_logits;
                    Eigen::VectorXf ent_g_logits=Eigen::VectorXf::Zero(config_.action_dim);
                    for(int k=0;k<config_.action_dim;++k)for(int j=0;j<config_.action_dim;++j){float d_p=c_probs[j]*((j==k?1.0f:0.0f)-c_probs[k]);ent_g_logits[k]+=-(std::log(c_probs[j]+1e-8f)+1.0f)*d_p;}
                    Eigen::VectorXf a_g_logits=plg_logits-config_.entropy_coeff*ent_g_logits;
                    actor_.backward(a_g_logits);for(size_t k=0;k<actor_.weights_.size();++k){b_agw[k]+=actor_.grad_weights_[k];b_agb[k]+=actor_.grad_biases_[k];}
                    Eigen::VectorXf c_vo=critic_.forward(tr.state);float c_v=c_vo[0];Eigen::VectorXf c_g_o=Eigen::VectorXf::Constant(1,2.0f*(c_v-tr.v_target));
                    critic_.backward(c_g_o);for(size_t k=0;k<critic_.weights_.size();++k){b_cgw[k]+=critic_.grad_weights_[k];b_cgb[k]+=critic_.grad_biases_[k];}
                }
                float mbsf=static_cast<float>(b_end-i);if(mbsf>0){
                    for(size_t k=0;k<actor_.weights_.size();++k){actor_.grad_weights_[k]=b_agw[k]/mbsf;actor_.grad_biases_[k]=b_agb[k]/mbsf;}actor_optimizer_.update(actor_);
                    for(size_t k=0;k<critic_.weights_.size();++k){critic_.grad_weights_[k]=b_cgw[k]/mbsf;critic_.grad_biases_[k]=b_cgb[k]/mbsf;}critic_optimizer_.update(critic_);
                }
            }
        }
        buf.clear();
    }
    bool save_model_params(godot::FileAccess* file) const {
        if(!actor_.save_parameters(file))return false;if(!critic_.save_parameters(file))return false;
        if(!actor_optimizer_.save_state(file))return false;if(!critic_optimizer_.save_state(file))return false;
        return true;
    }
    bool load_model_params(godot::FileAccess* file) {
        if(!actor_.load_parameters(file))return false;if(!critic_.load_parameters(file))return false;
        if(!actor_optimizer_.load_state(file,actor_))return false;if(!critic_optimizer_.load_state(file,critic_))return false;
        return true;
    }
};

} // Ende PPOInternal Namespace


PPO::PPO() {}
PPO::~PPO() {}

void PPO::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("initialize", "config"), &PPO::initialize);
    godot::ClassDB::bind_method(godot::D_METHOD("get_action", "observation_array"), &PPO::get_action);
    godot::ClassDB::bind_method(godot::D_METHOD("store_experience", "reward", "next_observation_array", "done"), &PPO::store_experience);
    godot::ClassDB::bind_method(godot::D_METHOD("train"), &PPO::train);
    godot::ClassDB::bind_method(godot::D_METHOD("save_model", "file_path"), &PPO::save_model);
    godot::ClassDB::bind_method(godot::D_METHOD("load_model", "file_path"), &PPO::load_model);
}

void PPO::initialize(const godot::Dictionary& config) {
    PPOInternal::PPOCoreConfig ppo_config;
    observation_dim_ = config.get("observation_dim", 0); action_dim_ = config.get("action_dim", 0);
    ppo_config.obs_dim = observation_dim_; ppo_config.action_dim = action_dim_;
    godot::Array actor_h_gd = config.get("actor_hidden_dims", godot::Array()); for(int i=0;i<actor_h_gd.size();++i)ppo_config.actor_hidden_dims.push_back(actor_h_gd[i]);
    godot::Array critic_h_gd = config.get("critic_hidden_dims", godot::Array()); for(int i=0;i<critic_h_gd.size();++i)ppo_config.critic_hidden_dims.push_back(critic_h_gd[i]);
    ppo_config.lr_actor = config.get("lr_actor", 0.0003f); ppo_config.lr_critic = config.get("lr_critic", 0.001f);
    ppo_config.gamma = config.get("gamma", 0.99f); ppo_config.lambda_gae = config.get("lambda_gae", 0.95f);
    ppo_config.clip_epsilon = config.get("clip_epsilon", 0.2f); ppo_config.ppo_epochs = config.get("ppo_epochs", 10);
    ppo_config.minibatch_size = config.get("minibatch_size", 64); ppo_config.entropy_coeff = config.get("entropy_coeff", 0.01f);
    ppo_config.seed = config.get("seed", (unsigned int)std::random_device{}());
    buffer_size_ = config.get("buffer_size", 2048); train_every_n_steps_ = config.get("train_every_n_steps", buffer_size_);
    if(observation_dim_<=0||action_dim_<=0||ppo_config.actor_hidden_dims.empty()||ppo_config.critic_hidden_dims.empty()){
        initialized_=false; return;
    }
    ppo_core_ = std::make_unique<PPOInternal::PPOCore>(ppo_config);
    replay_buffer_ = std::make_unique<PPOInternal::ReplayBuffer>(buffer_size_, ppo_config.seed + 3);
    initialized_ = true; training_counter_ = 0; current_observation_.resize(0);
}
int PPO::get_action(const godot::PackedFloat32Array& obs_arr) {
    if(!initialized_){return -1;}
    if(obs_arr.size()!=observation_dim_){return -1;}
    current_observation_ = packed_array_to_eigen(obs_arr);
    auto[act,lp,val_est] = ppo_core_->select_action_details(current_observation_);
    current_action_=act; current_action_log_prob_=lp; current_value_estimate_=val_est;
    return act;
}
void PPO::store_experience(float reward, const godot::PackedFloat32Array& next_obs_arr, bool done) {
    if(!initialized_){return;}
    if(next_obs_arr.size()!=observation_dim_){return;}
    if(current_observation_.size()!=observation_dim_){return;}
    Eigen::VectorXf next_obs=packed_array_to_eigen(next_obs_arr);
    replay_buffer_->add_transition(current_observation_,current_action_,reward,next_obs,done,current_action_log_prob_,current_value_estimate_);
    training_counter_++; current_observation_.resize(0);
    if(training_counter_>=train_every_n_steps_){train();training_counter_=0;}
}
void PPO::train() {
    if(!initialized_||replay_buffer_->size()==0)return;
    float last_val=0.0f;
    if(!replay_buffer_->buffer_.empty()){
        const auto& last_trans=replay_buffer_->buffer_.back();
        if(!last_trans.done)last_val=ppo_core_->critic_.forward(last_trans.next_state)[0];
    }
    ppo_core_->update(*replay_buffer_,last_val);
}
bool PPO::save_model(const godot::String& file_path) {
    if(!initialized_||!ppo_core_){return false;}
    godot::Error err; godot::Ref<godot::FileAccess>file=godot::FileAccess::open(file_path,godot::FileAccess::WRITE);
    if(file.is_null()||!file->is_open()){err=godot::FileAccess::get_open_error();return false;}
    file->store_string("PPOAMDL"); file->store_32(SAVE_FORMAT_VERSION);
    file->store_32(static_cast<uint32_t>(observation_dim_)); file->store_32(static_cast<uint32_t>(action_dim_));
    if(!ppo_core_->save_model_params(file.ptr())){file->close();return false;}
    err=file->get_error(); file->close();
    if(err!=godot::OK){return false;}
    return true;
}
bool PPO::load_model(const godot::String& file_path) {
    if(!initialized_||!ppo_core_){return false;}
    if(!godot::FileAccess::file_exists(file_path)){return false;}
    godot::Error err; godot::Ref<godot::FileAccess>file=godot::FileAccess::open(file_path,godot::FileAccess::READ);
    if(file.is_null()||!file->is_open()){err=godot::FileAccess::get_open_error();return false;}
    char magic_buf[8]={0}; uint64_t len_r=file->get_buffer((uint8_t*)magic_buf,7);
    if(len_r!=7||godot::String(magic_buf)!="PPOAMDL"){file->close();return false;}
    uint32_t file_v=file->get_32(); if(file_v>SAVE_FORMAT_VERSION){file->close();return false;}
    uint32_t s_obs_d=file->get_32(); uint32_t s_act_d=file->get_32();
    if(static_cast<int>(s_obs_d)!=observation_dim_||static_cast<int>(s_act_d)!=action_dim_){
        file->close();return false;
    }
    if(!ppo_core_->load_model_params(file.ptr())){file->close();return false;}
    err=file->get_error(); file->close();
    if(err!=godot::OK){return false;}
    return true;
}