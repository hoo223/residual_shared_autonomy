#!/usr/bin/python
# -*- coding: utf8 -*- 

if __name__ == '__main__':
    from residual_shared_autonomy.ur10_actors import UR10JoystickActor, UR10KeyboardActor # Joystick Agent
    import gym
    import numpy as np
    import rospy
    from dl.rl import ensure_vec_env
    
    prefix = 'unity'
    env = gym.make("ur10_env:ur10-v0")
    env = ensure_vec_env(env)
    actor = UR10KeyboardActor(action_mask=[1, 0, 0, 0, 0, 0], random=True)
    
    for _ in range(5):
        ob = env.reset()
        print(ob)
        done = False
        reward = 0.0
        step = 0
        while not done:
            step += 1
            action = actor(ob)
            print(action)
            ob, r, done, _ = env.step(action)
            reward += r
            if reward > 2000:
                done = True
            print("Step {} - reward: {}".format(step, reward))
    env.close()