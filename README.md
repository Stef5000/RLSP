Native reinforcement-learning algorithms for Godot!
-
Supported algorithms:

-PPO (Discrete) as ppo
-PPO (Continous) as ppoc


Runs only on the CPU but should be pretty fast.

Install Instructions:

-Grab a Release Build
-Extract the file
-Put the resulting folder anywhere in your Project. Done!

Example:

This Example will move an Agent to a goal with the discrete PPO Model
```
extends Node2D

@onready var agent: Sprite2D = $Agent
@onready var goal: Sprite2D = $Goal

var ppo: PPO
const AGENT_SPEED: float = 200.0
const MAX_EPISODE_STEPS: int = 400 
const GOAL_REACHED_RADIUS: float = 25.0

const OBSERVATION_DIM = 4 
const ACTION_DIM = 4

const BUFFER_SIZE = 1024
const TRAIN_EVERY_N_STEPS = 1024

var current_observation: PackedFloat32Array
var episode_count: int = 0
var steps_in_episode: int = 0
var total_reward_in_episode: float = 0.0

func _ready() -> void:
	ppo = PPO.new()
	ppo.initialize({
		"observation_dim":4,
		"action_dim":4,
		"actor_hidden_dims":[64,32],
		"critic_hidden_dims":[64, 32],
		"lr_actor":0.003,
		"lr_critic":0.001,
		"gamma":0.99,
		"lamda_gae":0.95,
		"clip_epsilon": 0.2,
	})
	start_new_episode()

func start_new_episode():
	steps_in_episode = 0
	total_reward_in_episode = 0
	episode_count += 1
	var viewport_size = get_viewport_rect().size
	goal.position = Vector2(
		randf_range(50, viewport_size.x - 50),
		randf_range(50, viewport_size.y - 50)
	)
	agent.position = viewport_size / 2.0
	current_observation = get_observation()
	print("Started Episode #", episode_count)

func get_observation() -> PackedFloat32Array:
	var viewport_size = get_viewport_rect().size
	var vector_to_goal = goal.position - agent.position
	var obs = PackedFloat32Array()
	obs.resize(OBSERVATION_DIM)
	obs[0] = agent.position.x / viewport_size.x
	obs[1] = agent.position.y / viewport_size.y
	obs[2] = vector_to_goal.x / viewport_size.x
	obs[3] = vector_to_goal.y / viewport_size.y
	return obs

func _physics_process(delta: float) -> void:
	var action_index: int = ppo.get_action(current_observation)
	var last_distance_to_goal = agent.position.distance_to(goal.position)
	var move_vector = Vector2.ZERO
	match action_index:
		0: move_vector = Vector2.UP
		1: move_vector = Vector2.DOWN
		2: move_vector = Vector2.LEFT
		3: move_vector = Vector2.RIGHT
	agent.position += move_vector * AGENT_SPEED * delta
	var viewport_size := get_viewport_rect().size
	agent.position.x = clamp(agent.position.x, 0, viewport_size.x)
	agent.position.y = clamp(agent.position.y, 0, viewport_size.y)
	var new_distance_to_goal = agent.position.distance_to(goal.position)
	var reward: float = 0.0
	var done: bool = false
	reward = (last_distance_to_goal - new_distance_to_goal) * 0.1
	if new_distance_to_goal < GOAL_REACHED_RADIUS:
		reward += 100.0
		done = true
		print("Goal Reached! Reward: ", total_reward_in_episode + reward)
	reward -= 0.1
	steps_in_episode += 1
	if steps_in_episode >= MAX_EPISODE_STEPS:
		done = true
		reward -= 20.0
		print("Time limit reached. Reward: ", total_reward_in_episode + reward)
	total_reward_in_episode += reward
	var next_observation = get_observation()
	ppo.store_experience(reward, next_observation, done)
	current_observation = next_observation
	if done:
		start_new_episode()
```
