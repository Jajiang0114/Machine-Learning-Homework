# Playing with OpenAI Gym: CartPole-v0

import time
import gym
import numpy as np
import random
from sklearn.neural_network import MLPClassifier

##################################################################################################
# policies

def naive_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

def random_policy(obs):
	return random.randrange(0,3)

def my_policy(obs):
	move = 1
	# If no previous observation or at starting location make a move based on position
	if (obs[1] == 0):
		if(obs[0] <= 0):
			move = 2
		elif(obs[0] > 0):
			move = 0
		return move
	#If on the left hill
	if(obs[0] < 0):
		if(obs[1] < 0):
			move = 0
		else:
			move = 2
	elif (obs[0] > 0):
		if(obs[1] > 0):
			move = 2
		else:
			move = 0

	else:
		move = 1
	return move
	
def NN_policy(obs):
	mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
	X,Y = get_data()
	#print(X)
	#print(Y)
	mlp.fit(X, Y)
	move = mlp.predict([obs])
	#print(move)
	return move[0]

##################################################################################################


def get_data():
	#print("Gathering Test Data")
	env = gym.make("MountainCar-v0")
	env.reset()
	training_X = []
	training_Y = []
	while (training_X == [] or training_Y == []):
		for episode in range(100):
			score = 0

			mem = []
			prev_obs = []

			for steps in range(10000):
				action = random.randrange(0,3)
				
				obs, reward, done, info = env.step(action)
				
				if len(prev_obs) > 0 :
					mem.append([prev_obs, action])
				if (prev_obs != []):
					if (obs[0] <0):
						if((obs[1] < prev_obs[1]) and (obs[0] < prev_obs[0])):
							score += 1
					if (obs[0] > 0):
						if((obs[1] > prev_obs[1]) and (obs[0] > prev_obs[0])):
							score += 1
				prev_obs = obs
				#score+=reward
				if done: break

			if score >= 50:

				for data in mem:
					training_X.append(data[0])
					training_Y.append(data[1])
			env.reset()

	return training_X, training_Y

def naive_main( policy ):
	debug = True
	env = gym.make("MountainCar-v0")
	obs = env.reset()
	env.render()

	# episodic reinforcement learning
	totals = []
	for episode in range(100):
		episode_rewards = 0
		obs = env.reset()
		#prev_obs = []
		for step in range(10000):
			action = policy(obs)
			obs, reward, done, info = env.step(action)
			env.render()
			time.sleep(0.1)
			episode_rewards += reward
			#prev_obs = obs
			if done:
				print ("Game over. Number of steps = ", step)
				env.render()
				time.sleep(3.14)
				break
		totals.append(episode_rewards)
		print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

##################################################################################################

if __name__ == "__main__":
	naive_main( my_policy )

##################################################################################################

