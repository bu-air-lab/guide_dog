import gym

#We need to pass an environment into OnPolicyRunner to load our policy for some reason
class BlankEnv(gym.Env):

    def __init__(self, use_force_estimator=True):

        self.num_privileged_obs = None
        self.num_obs = 48 #120
        self.num_envs = 1
        self.num_actions = 12
        self.use_force_estimator = use_force_estimator
        self.isBlankEnv = True

    def reset(self):

        return None, None



