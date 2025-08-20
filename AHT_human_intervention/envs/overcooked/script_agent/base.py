class BaseScriptPeriod:
    """A base class for script period."""

    def __init__(self, period_name):
        self.period_name = period_name

    def reset(self, mdp, state, player_idx):
        """reset some script period"""
        raise NotImplementedError

    def step(self, mdp, state, player_idx):
        raise NotImplementedError

    def done(self, mdp, state, player_idx):
        raise NotImplementedError


class BaseScriptAgent:
    """A script agent consists of several script periods."""

    def __init__(self):
        self.agent_index = None
        self.mdp = None

    def reset(self, mdp, state, player_idx):
        """reset state"""
        self.agent_index = None
        self.mdp = None

    def step(self, mdp, state, player_idx):
        raise NotImplementedError

    def set_agent_index(self, agent_index):
        """Set the agent's index (0 for ego, 1 for confederate)"""
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        """Set the MDP for the agent"""
        self.mdp = mdp

    def action(self, state):
        """Interface method required by the environment"""
        return self.step(self.mdp, state, self.agent_index)
