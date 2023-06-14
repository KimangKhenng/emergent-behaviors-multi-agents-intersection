class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.front_views = []
        self.next_front_views = []
        self.rewards = []
        self.logprobs = []
        self.state_action_values = []
        self.is_terminated = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_terminated(self, terminated):
        self.is_terminated.append(terminated)

    def add_actions(self, actions):
        self.actions.append(actions)

    def add_state(self, state):
        self.states.append(state)

    def add_front_view(self, front_view):
        self.front_views.extend(front_view)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.front_views[:]
        del self.rewards[:]
        del self.state_action_values[:]
        del self.is_terminated[:]
        del self.logprobs[:]
        del self.next_states[:]