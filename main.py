import os
from stable_baseline3 import PPO
from stable_basline3.vec_env import DummyVecEnv
from stable_baseline3.common.evalutaion import evaluation_policy

"""

PPO is Proximal Policiy Optimization algorithm that combines ideas from A2C(having multiple workers) and TRPO(it uses a trust region to improve the actor)

A2C is Advantage Actor Critic algorithm that combines two types of RL algorithms (Policy Based and Value Based) together. what it does is mapping input states to output actions

TRPO is Trust Region Policy Optimization. it is a policy gradient method in reinforcement learning that avoids parameters updates that change the policy too much with a KL divergence

KL is Kullback-Leibler divergence (hereafter written as KL divergence) is a measure of how a probability distribution differs from another probability distribution. 

"""

env_name = "CartPole-V0"
env = gym.make(env_name)

episodes = 5

for episode in range(0, episodes+1):
	state = env.reset() #gives back array of contains initial values for our environment
	done = False
	score = 0

	while not done:
		env.render()
		action = env.action_space.sample() #generates a random number(0,1)
		n_state, reward, done, info = env.step(action) #env.step(n) returns array of reward, boolean state, info

		
		score += reward
		print(f'Episode: {episode}, Score: {score}')



#saving the Training logs
log_path = os.path.join("Training","Logs")


#use the DummyVec for env and traing model using MlpPolicy
#MlpPolicy is Multi Layer Perceptron Policy generates a set of outputs from a set of inputs MLP uses backpropogation for training the network.

env = DummyVecEnv([lambda:env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.train(total_timesteps=20000)


#save the model and delete it. load it whenver you want to run it

PPO_Path = os.path.join("Training","Saved Models", "PPO_Model_CartPole")
model.save(PPO_Path)
del model #deleting the model
model = PPO.load(PPO_Path, env=env) #loading the saved model

evaluate_policy(model, env, n_eval_episodes=10, render=True) #to test and evaluate performance


#now let's test model by letting it take its observations and use it to predict the results. that's what RL is about

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.step(n) returns array, reward, boolean state, info
        
        score += reward
        print(f'Episodes:{episode}, Score:{score}')
env.close()

