#!/usr/bin/python
# -*- coding: utf8 -*- 

if __name__ == '__main__':
    from rsa.ur10_actors import UR10JoystickActor # Joystick Agent
    import gym
    import numpy as np
    import rospy
    from dl.rl import ensure_vec_env
    
    prefix = 'unity'
    env = gym.make("ur10_env:ur10-v0")
    env = ensure_vec_env(env)
    actor = UR10JoystickActor(action_mask=[1, 1, 1, 0, 0, 0], random=True)
    
    for _ in range(5):
        ob = env.reset()
        done = False
        reward = 0.0

        while not done:
            action = actor(ob)
            print(action)
            ob, r, done, _ = env.step(action)
            reward += r
            if reward > 2000:
                done = True
            print(reward)
    env.close()