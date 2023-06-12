import torch
import torch.nn as nn
import torch.optim as optim
from mcts import GenericMCTS, A2CNode


class A2CAgent(nn.Module):
    def __init__(self, input_size, action_space, hidden_size=128, gamma=0.99, lr=1e-3, num_simulations=100):
        super(A2CAgent, self).__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.last_move = None

        self.gamma = gamma
        self.action_space = action_space
        self.num_simulations = num_simulations

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

    def act(self, state, action_space):
        action = self.select_action_with_mcts(state, action_space)
        self.last_move = action
        print(f"{self.__class__.__name__} last_move: {self.last_move}")
        return action

    def select_action_with_mcts(self, state, action_space):
        state = torch.FloatTensor(state).to(self.device)  # Pass 'state' instead of 'state.board'
        legal_actions = action_space
        root = A2CNode(state, legal_actions)
        mcts = GenericMCTS(self, action_space, root)
        best_action = mcts.run(len(state), self.num_simulations, action_space)
        return best_action

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)

        _, value = self.forward(state)
        _, next_value = self.forward(next_state)

        advantage = reward + (1 - done) * self.gamma * next_value - value

        policy, _ = self.forward(state)
        log_prob = torch.log(policy.squeeze(0)[action])

        loss = (-log_prob * advantage + 0.5 * advantage.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.critic.state_dict(), filename + "_critic.pt")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
