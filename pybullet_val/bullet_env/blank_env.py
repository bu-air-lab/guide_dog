import gym

#We need to pass an environment into OnPolicyRunner to load our policy for some reason
class BlankEnv(gym.Env):

    def __init__(self):

        self.num_privileged_obs = None
        self.num_obs = 117 #114 #186 #42
        self.num_envs = 1
        self.num_actions = 12

    def reset(self):

        return None, None



