Native reinforcement-learning algorithms for Godot!
-
Supported algorithms:

-PPO (Discrete, CPU) as PPO

-PPO (Continous, CPU) as PPOC

-DQN (Discrete) as DQN


Install Instructions:

-Grab a Release Build
-Extract the file
-Put the resulting folder anywhere in your Project. Done!

Examples:
This Example uses the DQN Algorithm to move a point towards a target

```
extends Node2D

@onready var p := $P
@onready var g := $G
var agent : DQN
const mult : int = 20

func _ready() -> void:
	agent = DQN.new()
	agent.initialize(2,4,0.0003,64,0.99,0.01,64,32)
	var path = ProjectSettings.globalize_path("res://model_data//modelpunkt.dat")
	agent.load_model(path)

func _physics_process(_delta: float) -> void:
	var obs : PackedFloat32Array = get_observation()
	var pp : Vector2 = p.position
	var ep_done : bool = false
	var reward : float = 0.0
	var action = agent.get_action(obs)
	match action:
		0: p.position.x += 10
		1: p.position.x -= 10
		2: p.position.y += 10
		3: p.position.y -= 10
	var np : Vector2 = p.position
	reward += (pp.distance_to(g.position)-np.distance_to(g.position)) * 0.1
	if p.position.y > 700 or p.position.y < 0 or p.position.x < 0 or p.position.x > 1100: 
		reward -= 100; ep_done = true
		p.position = Vector2(200,300)
	if p.position.distance_to(g.position) < 20:
		reward += 100; ep_done = true
		p.position = Vector2(200,300)
		g.position = Vector2(randi_range(100,1000),randi_range(50,600))
	var new_obs = get_observation()
	agent.add_experience(obs,action,reward,new_obs,ep_done)
	agent.train()
	if ep_done:
		agent.end_episode()

func get_observation():
	var max_x = 1100.0
	var max_y = 700.0
	var dx = (g.position.x - p.position.x) / max_x
	var dy = (g.position.y - p.position.y) / max_y
	var obs := PackedFloat32Array([dx,dy])
	return obs

func _unhandled_input(_event: InputEvent) -> void:
	if Input.is_action_just_pressed("speed"):
		if Engine.physics_ticks_per_second == 60 * mult:
			Engine.physics_ticks_per_second = 60
			Engine.max_physics_steps_per_frame = 8
		else:
			Engine.physics_ticks_per_second = 60 * mult
			Engine.max_physics_steps_per_frame = 32
	if Input.is_action_just_pressed("save"):
		var path = ProjectSettings.globalize_path("res://model_data//modelname.dat")
		agent.save_model(path)
```

This Example uses the Continous PPO algorithm to achieve the same
```
extends Node2D

@onready var p := $P
@onready var g := $G
var agent : PPOC
const mult : int = 20

func _ready() -> void:
	agent = PPOC.new()
	var config = {
		"observation_dim": get_observation().size(),
		"action_dim": 2,
		"actor_hidden_dims": [16, 16], 
		"critic_hidden_dims": [16, 16],  
		"lr_actor": 0.0003,
		"lr_critic": 0.001,
		"gamma": 0.99,
		"lambda_gae": 0.95,
		"clip_epsilon": 0.2,
		"ppo_epochs": 10,         
		"minibatch_size": 64,    
		"entropy_coeff": 0.01,
		"buffer_size": 256,
		"train_every_n_steps": 256 
	}
	agent.initialize(config)
	var path = ProjectSettings.globalize_path("res://model_data//training1.dat")
	agent.load_model(path)

func _physics_process(_delta: float) -> void:
	var obs : PackedFloat32Array = get_observation()
	var pp : Vector2 = p.position
	var ep_done : bool = false
	var reward : float = 0.0
	var action = agent.get_action(obs)
	print(action)
	p.position += Vector2(action[0],action[1])
	var np : Vector2 = p.position
	reward += (pp.distance_to(g.position)-np.distance_to(g.position)) * 0.1
	if p.position.y > 700 or p.position.y < 0 or p.position.x < 0 or p.position.x > 1100: 
		reward -= 100; ep_done = true
		p.position = Vector2(200,300)
	if p.position.distance_to(g.position) < 20:
		reward += 100; ep_done = true
		p.position = Vector2(200,300)
		g.position = Vector2(randi_range(100,1000),randi_range(50,600))
	var new_obs = get_observation()
	agent.store_experience(reward,new_obs,ep_done)
	agent.train()

func get_observation():
	var max_x = 1100.0
	var max_y = 700.0
	var dx = (g.position.x - p.position.x) / max_x
	var dy = (g.position.y - p.position.y) / max_y
	var obs := PackedFloat32Array([dx,dy])
	return obs

func _unhandled_input(_event: InputEvent) -> void:
	if Input.is_action_just_pressed("change_speed"):
		if Engine.physics_ticks_per_second == 60 * mult:
			Engine.physics_ticks_per_second = 60
			Engine.max_physics_steps_per_frame = 8
		else:
			Engine.physics_ticks_per_second = 60 * mult
			Engine.max_physics_steps_per_frame = 32
	if Input.is_action_just_pressed("save_model"):
		var path = ProjectSettings.globalize_path("res://model_data//training1.dat")
		agent.save_model(path)

```
