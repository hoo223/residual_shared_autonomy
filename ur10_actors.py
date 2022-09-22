#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 

"""Base actors on which residuals are learned."""
import numpy as np
import torch
import random as rnd
import time, pygame
import gin
import sys

#####################################
# Change these to match your joystick
RIGHT_UP_AXIS = 4
RIGHT_SIDE_AXIS = 3
LEFT_UP_AXIS = 1
LEFT_SIDE_AXIS = 0
#####################################

@gin.configurable
class UR10HumanActor(object):
    """Joystick Controller for UR10."""

    def __init__(self, env, action_mask=[1, 1, 1, 1, 1, 1], input_type='joystick'):
        """Init."""
        self.action_mask = action_mask
        self.action_scale = 1.0
        self.input_type = input_type
        self.human_agent_action = np.zeros(6)
        self.button = np.zeros(1)
        self.batch_size = 1 # num_envs
        
        if self.input_type == 'joystick':
            pygame.joystick.init()
            joysticks = [pygame.joystick.Joystick(x)
                        for x in range(pygame.joystick.get_count())]
            if len(joysticks) > 1:
                raise ValueError("There must be exactly 1 joystick connected.",
                                "Found ", len(joysticks))
            elif len(joysticks) == 0:
                raise ValueError("There is no joystick connected.")
            elif len(joysticks) == 1:   
                self.joy = joysticks[0]
                self.joy.init()
        elif self.input_type == 'keyboard':
            width, height = 640, 480
            screen = pygame.display.set_mode((width, height))
            
        pygame.init()
        self.t = None
        
    def _get_joystick_action(self):
        for event in pygame.event.get():
            # Joystick input
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == LEFT_SIDE_AXIS:
                    self.human_agent_action[1] = event.value
                elif event.axis == LEFT_UP_AXIS:
                    self.human_agent_action[0] = -1.0 * event.value
                if event.axis == RIGHT_SIDE_AXIS:
                    self.human_agent_action[5] = event.value
                elif event.axis == RIGHT_UP_AXIS:
                    self.human_agent_action[2] = -1.0 * event.value
            if event.type == pygame.JOYBUTTONDOWN:
                self.button[0] = event.button
                if self.button[0] == 1:
                    self.human_agent_action[3] = 1
                elif self.button[0] == 2:
                    self.human_agent_action[3] = -1
                if self.button[0] == 0:
                    self.human_agent_action[4] = 1
                elif self.button[0] == 3:
                    self.human_agent_action[4] = -1
            else: # button clear
                self.button[0] = -1
                self.human_agent_action[3] = self.human_agent_action[4] = 0
        #self.human_agent_action[0] = -1
        action = [self.human_agent_action[i] * self.action_mask[i] * self.action_scale for i in range(1)]
        self.action = [action for _ in range(self.batch_size)]
        return self.action
    
    def _get_keyboard_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.human_agent_action[0] = 1.0
        elif keys[pygame.K_s]:
            self.human_agent_action[0] = -1.0
        else:
            self.human_agent_action[0] = 0.0
            
        if keys[pygame.K_a]:
            self.human_agent_action[1] = -1.0
        elif keys[pygame.K_d]:
            self.human_agent_action[1] = 1.0
        else:
            self.human_agent_action[1] = 0.0
            
        if keys[pygame.K_e]:
            self.human_agent_action[2] = 1.0
        elif keys[pygame.K_c]:
            self.human_agent_action[2] = -1.0
        else:
            self.human_agent_action[2] = 0.0
            
        if keys[pygame.K_f]:
            self.human_agent_action[3] = 1.0
        elif keys[pygame.K_h]:
            self.human_agent_action[3] = -1.0
        else:
            self.human_agent_action[3] = 0.0
            
        if keys[pygame.K_t]:
            self.human_agent_action[4] = 1.0
        elif keys[pygame.K_g]:
            self.human_agent_action[4] = -1.0
        else:
            self.human_agent_action[4] = 0.0
            
        if keys[pygame.K_r]:
            self.human_agent_action[5] = 1.0
        elif keys[pygame.K_y]:
            self.human_agent_action[5] = -1.0
        else:
            self.human_agent_action[5] = 0.0
            
        if keys[pygame.K_1]:
            self.button[0] = 6.0 # init mode
        elif keys[pygame.K_2]:
            self.button[0] = 7.0 # teleop mode
        else:
            self.button[0] = -1
            
        action = [self.human_agent_action[i] * self.action_mask[i] * self.action_scale for i in range(1)]
        self.action = [action for _ in range(self.batch_size)]
        return self.action

    def __call__(self, ob):
        """Act."""
        if self.input_type == 'joystick':
            action = self._get_joystick_action()
        elif self.input_type == 'keyboard':
            action = self._get_keyboard_action()
        self.t = time.time()
        # Prevent collision with table
        if ob[0][2] < 0.025:
            if action[2] < 0:
                action[2] = 0
        return np.asarray(action)

    def reset(self):
        self.human_agent_action[:] = 0.

@gin.configurable
class UR10RandomActor(object):
    """Joystick Controller for UR10."""

    def __init__(self, env, action_mask=[1, 1, 1, 1, 1, 1]):
        """Init."""
        self.env = env
        self.action_mask = action_mask
        self.rnd = rnd
        self.rnd.seed(0)
        self.action_period = 10
        self.action_cnt = self.action_period
        self.batch_size = 1 # num_envs

    def _get_random_action(self):
        self.action_cnt += 1
        if self.action_cnt > self.action_period:
            action = [self.rnd.choice([-1, 0, 1])*self.action_mask[i] for i in range(1)]
            self.action = [action for _ in range(self.batch_size)]
            #self.action = [self.env.action_space.sample() for _ in range(self.batch_size)]
            self.action_cnt = 0
            self.action_period = rnd.randrange(5,101)
        return self.action

    def __call__(self, ob):
        """Act."""
        action = self._get_random_action()
        self.t = time.time()
        # Prevent collision with table
        if ob[0][2] < 0.025:
            if action[2] < 0:
                action[2] = 0
        return np.asarray(action)

    def reset(self):
        pass


if __name__ == '__main__':
    import gym
    from dl.rl import ensure_vec_env
    import time

    env = gym.make("ur10_env:ur10-v0")
    #env = ensure_vec_env(env)

    actor = UR10RandomActor()

    for _ in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0
        time.sleep(1.)

        while not done:
            ob, r, done, _ = env.step(actor(ob))
            env.render()
            reward += r
        print(reward)
