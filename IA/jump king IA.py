
#Il m'est impossible de mettre l'intégralité du jeu sur github car il prend trop de place. Voici donc le fichier à ajouter au github suivant : 
# https://github.com/senweim/JumpKingAtHome


#!/usr/env/bin python
#   
# Game Screen
# 
import os

# Set the environment variable to run Pygame in headless mode
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame 
import sys
import os
import inspect
import pickle
import numpy as np
from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus

from Start import Start

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda")



class NETWORK(torch.nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:

		super(NETWORK, self).__init__()

		self.layer1 = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.layer2 = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.final = torch.nn.Linear(hidden_dim, output_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.final(x)

		return x


class DDQN(object):
    def __init__(self):


        self.target_net = NETWORK(4, 4, 32).to(device)
        self.eval_net = NETWORK(4, 4, 32).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        self.memory_counter = 0
        self.memory_size = 100000
        self.memory = np.zeros((self.memory_size, 11))

        self.epsilon = 0.65
        self.epsilon_decay = 0.999
        self.alpha = 0.99

        self.batch_size = 2000
        self.episode_counter = 0

        self.target_net.load_state_dict(self.eval_net.state_dict())
		
    def memory_store(self, s0, a0, r, s1, sign):
        transition = np.concatenate((s0, [a0, r], s1, [sign]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def select_action(self, states: np.ndarray) -> int:
        state = torch.unsqueeze(torch.tensor(states).float(), 0).to(device)
        if np.random.uniform() > self.epsilon:
            logit = self.eval_net(state)
            action = torch.argmax(logit, 1).item()
        else:
            action = int(np.random.choice(4, 1))
        return action

    def policy(self, states: np.ndarray) -> int:
        state = torch.unsqueeze(torch.tensor(states).float(), 0).to(device)
        logit = self.eval_net(state)
        action = torch.argmax(logit, 1).item()
        return action

    def train(self, s0, a0, r, s1, sign):
        if sign == 1:
            if self.episode_counter % 2 == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.episode_counter += 1

        self.memory_store(s0, a0, r, s1, sign)
        self.epsilon = np.clip(self.epsilon * self.epsilon_decay, a_min=0.01, a_max=None)

        # select batch sample
        if self.memory_counter > self.memory_size:
            batch_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[batch_index]
        batch_s0 = torch.tensor(batch_memory[:, :4]).float().to(device)
        batch_a0 = torch.tensor(batch_memory[:, 4:5]).long().to(device)
        batch_r = torch.tensor(batch_memory[:, 5:6]).float().to(device)
        batch_s1 = torch.tensor(batch_memory[:, 6:10]).float().to(device)
        batch_sign = torch.tensor(batch_memory[:, 10:11]).long().to(device)

        q_eval = self.eval_net(batch_s0).gather(1, batch_a0)

        with torch.no_grad():
            maxAction = torch.argmax(self.eval_net(batch_s1), 1, keepdim=True)
            q_target = batch_r + (1 - batch_sign) * self.alpha * self.target_net(batch_s1).gather(1, maxAction)

        loss = self.criterion(q_eval, q_target)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, max_step=float('inf')):

		pygame.init()
		
		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = 1 *int(os.environ.get("fps"))
 
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

		self.levels = Levels(self.game_screen)

		self.king = King(self.game_screen, self.levels)
	
		self.babe = Babe(self.game_screen, self.levels)

		self.menus = Menus(self.game_screen, self.levels, self.king)

		self.start = Start(self.game_screen, self.menus)

		self.step_counter = 0
		self.max_step = max_step

		self.visited = {}

		pygame.display.set_caption('Jump King AI')

	def reset(self):

		self.king.reset()
		self.levels.reset()
		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"
		self.epsilon = 0.001
		self.step_counter = 0
		
		self.level_up_reward_given1 = False
		self.level_up_reward_given2 = False
		self.level_up_reward_given3 = False
		self.level_up_reward_given4 = False
		self.level_up_reward_given2 = False
		self.level_up_reward_given5 = False
		self.level_up_reward_given6 = False
		self.level_up_reward_given7 = False
		self.level_up_reward_given8 = False
		self.max_level = 0
		self.max_y = 297
		
		done = False
		state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]

		self.visited = {}
		self.visited[(self.king.levels.current_level, self.king.y)] = 1
		self.max_level = 0
		self.max_y = 297
		return done, state

	def move_available(self):
		available = not self.king.isFalling \
					and not self.king.levels.ending \
					and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
		return available

	def step(self, action):
		old_level = self.king.levels.current_level
		old_y = self.king.y
		old_x = self.king.x
	
		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
		while True:
			self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				if not self.move_available():
					action = None
				self._update_gamestuff(action=action)

			self._update_gamescreen()
			#self._update_guistuff()
			#self._update_audio()
			pygame.display.update()

			
			if self.move_available():
				self.step_counter += 1
				state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
		
				##################################################################################################
				# Define the reward from environment                                                             #
				##################################################################################################
			
				if self.king.levels.current_level == self.max_level and self.king.y < self.max_y:
					if self.max_y - self.king.y > 3 :
						
						self.max_y = self.king.y
						
						reward = 50 
						print("new max y")
						print(self.max_y)

					else :
						reward = 0
				
				elif self.king.levels.current_level == self.max_level and self.king.y == self.max_y and self.king.y != 297:
					reward = 0.0001 * ((1) * (300 - self.max_y)) + (0.0001 * self.king.levels.current_level)
					
					#print("dora")
	
		
				elif self.king.levels.current_level == 1 and self.level_up_reward_given1 == False :
					reward = 0
					self.max_level = 1
				
					self.max_y = 300
					print("new max level")
					
					print("y  =" + str(self.king.y))
					self.level_up_reward_given1 = True

				elif self.king.levels.current_level == 2 and self.level_up_reward_given2 == False :
					reward = 0
					self.max_level = 2
				
					self.max_y = 300
		
				
				
					self.level_up_reward_given2 = True
				elif self.king.levels.current_level == 3 and self.level_up_reward_given3 == False :
					reward = 0
					self.max_level = 3
				
					self.max_y = 300
			

					self.level_up_reward_given3 = True
				elif self.king.levels.current_level == 4 and self.level_up_reward_given4 == False :
					reward = 0
					self.max_level = 4
				
					self.max_y = 300
			

					self.level_up_reward_given4 = True


				elif (self.king.levels.current_level == old_level and self.king.y < old_y ):
					reward = 20	
				#print("MORE =" + str(reward))
				#print("y  =" + str(self.king.y))
				elif (self.king.levels.current_level < old_level):
        		# Penalize descendre
					print("ahahah")
					reward = 0
				#	done = True 

				#	return state, reward, done

				elif (self.king.levels.current_level == old_level and self.king.y > old_y ):
        		# Penalize descendre
					reward = 0
					print("noooooo")
				#	done = True 

				#	return state, reward, done
			

				else:
					
					reward = 0
					
	


				if self.step_counter == 1 : 
					self.max_y = 297
				done = True if self.step_counter > self.max_step else False

				return state, reward, done


	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		
		print("reset")
		while True:
			state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
			print(state)
			self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				self._update_gamestuff()

			self._update_gamescreen()
			self._update_guistuff()
			#self._update_audio()
			pygame.display.update()

	def _check_events(self):

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				self.environment.save()

				self.menus.save()

				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self, action=None):

		self.levels.update_levels(self.king, self.babe, agentCommand=action)

	def _update_guistuff(self):

		if self.menus.current_menu:

			self.menus.update()

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King not by me - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			self.king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		self.menus.blitme()

		self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

	def _resize_screen(self, w, h):

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

	def _shake_screen(self):

		try:

			if self.levels.levels[self.levels.current_level].shake:

				if self.levels.shake_var <= 150:

					self.game_screen_x = 0

				elif self.levels.shake_var // 8 % 2 == 1:

					self.game_screen_x = -1

				elif self.levels.shake_var // 8 % 2 == 0:

					self.game_screen_x = 1

			if self.levels.shake_var > 260:

				self.levels.shake_var = 0

			self.levels.shake_var += 1

		except Exception as e:

			print("SHAKE ERROR: ", e)

	def _update_audio(self):

		for channel in range(pygame.mixer.get_num_channels()):

			if not os.environ["music"]:

				if channel in range(0, 2):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["ambience"]:

				if channel in range(2, 7):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["sfx"]:

				if channel in range(7, 16):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))

def train():
		

    best_max_level = 0
    best_maxy = 300
    current_maxy = 300
    current_max_level = 0
    action_dict = {
        0: 'right',
        1: 'left',
        2: 'right+space',
        3: 'left+space',
        # 4: 'idle',
        # 5: 'space',
    }

    agent = DDQN()
    env = JKGame(max_step=250)
    num_episode = 1000000
    save_interval = 20  # Sauvegarder toutes les 10 épisodes
    save_directory = r'C:\Users\WC272\Desktop\CESI2\veille\save\v16'

    # Charger le dernier modèle sauvegardé s'il existe
    new_model = True
    latest_model_num = 11721

    latest_model = r'C:\Users\WC272\Desktop\CESI2\veille\save\v14\modelv33_episode_10121.pth'
    if new_model == False:
        if latest_model:
            agent.eval_net.load_state_dict(torch.load(os.path.join(save_directory, latest_model)))
    else:
        latest_model_num = 0

    for i in range(num_episode):
        done, state = env.reset()
        running_reward = 0

        while not done:
            current_maxy = 300
            current_max_level = 0
            action = agent.select_action(state)

            next_state, reward, done = env.step(action)
            running_reward += reward
            sign = 1 if done else 0
            agent.train(state, action, reward, next_state, sign)
            state = next_state

        print(f'Episode: {latest_model_num + i}, Reward: {running_reward}')
		        # Update the best max_y and max_level
        
        if best_max_level < current_max_level :
           best_max_level = current_max_level
           best_maxy = 300
           print("ahaahahahahh")
        elif best_max_level  == current_max_level and best_maxy > current_maxy :
            best_maxy = current_maxy
            print("bbabababba")
        # Sauvegarder le modèle
        if i % save_interval == 0:
            print('SAVED')
            save_path = os.path.join(save_directory, f'modelv40_episode_{latest_model_num + i + 1}.pth')
            torch.save(agent.eval_net.state_dict(), save_path)



def best_run(model_path, env):
    agent = DDQN()
    agent.eval_net.load_state_dict(torch.load(model_path))
    agent.eval_net.eval()  # Mettre le réseau en mode évaluation

    num_episodes = 1  # Nombre d'épisodes pour l'exploitation
    for _ in range(num_episodes):
        done, state = env.reset()
        total_reward = 0

        while not done:
            action = agent.policy(state)  # Utiliser la politique sans exploration
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print(f'Episode: {_ + 1}, Total Reward: {total_reward}')


if __name__ == "__main__":
    device = torch.device("cuda")
    print(torch.__version__)
    env = JKGame(max_step=150)

    model_path = r'C:\Users\WC272\Desktop\CESI2\veille\save\v12\modelv13_episode_11721.pth'

    best_run(model_path, env)

    #Game = JKGame()
    # Game.running()
    #train()
