import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque
import random
import torch
import os

STUDENT_ID = 'a1234567'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # Define state dimensions
        self.state_dim = 5  # For our state representation (see build_state method)
        self.action_dim = 2  # Jump or do nothing

        # Hyperparameters
        self.epsilon = 0.1  # Starting exploration rate
        self.epsilon_decay = 0.999  # Rate at which epsilon decays
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.discount_factor = 0.99  # Gamma in Algorithm 2
        self.n = 64  # Batch size for training
        self.memory_size = 10000  # Size of replay buffer
        self.update_target_freq = 10  # How often to update target network

        # Storage for experience replay (D in Algorithm 2)
        self.storage = deque(maxlen=self.memory_size)
        
        # Learning parameters
        self.learning_rate = 0.001
        
        # Setup networks
        self.network = MLPRegression(
            input_dim=self.state_dim, 
            output_dim=self.action_dim, 
            hidden_dim=[200, 500, 100],
            learning_rate=self.learning_rate
        )
        
        # Target network for stable learning
        self.network2 = MLPRegression(
            input_dim=self.state_dim, 
            output_dim=self.action_dim, 
            hidden_dim=[200, 500, 100],
            learning_rate=self.learning_rate
        )
        
        # Initialize target network with same weights as main network
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)
        
        # Counter for updating target network
        self.training_steps = 0
        
        # Store previous state and action
        self.prev_state = None
        self.prev_action = None
        
        # Track game stats for reward calculation
        self.last_score = 0
        self.last_mileage = 0
        
        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def build_state(self, state_dict):
        """
        Convert the game state dictionary to a feature vector suitable for the neural network.
        
        Extracts relevant features that the agent needs to learn:
        - Bird's vertical position (normalized)
        - Bird's vertical velocity (normalized)
        - Distance to next pipe (normalized)
        - Height of top pipe (normalized)
        - Height of bottom pipe (normalized)
        """
        bird_y = state_dict['bird_y'] / state_dict['screen_height']
        bird_velocity = state_dict['bird_velocity'] / 10.0  # Normalize velocity
        
        # Find the next pipe (first pipe to the right of the bird)
        next_pipe = None
        min_distance = float('inf')
        
        for pipe in state_dict['pipes']:
            # If the pipe is ahead of the bird
            if pipe['x'] + state_dict['pipe_attributes']['width'] > state_dict['bird_x']:
                distance = pipe['x'] - state_dict['bird_x']
                if distance < min_distance:
                    min_distance = distance
                    next_pipe = pipe
        
        # If there are no pipes ahead, use default values
        if next_pipe is None:
            pipe_distance = 1.0
            top_pipe_height = 0.5
            bottom_pipe_height = 0.5
        else:
            pipe_distance = (next_pipe['x'] - state_dict['bird_x']) / state_dict['screen_width']
            top_pipe_height = next_pipe['top'] / state_dict['screen_height']
            bottom_pipe_height = next_pipe['bottom'] / state_dict['screen_height']
        
        return np.array([bird_y, bird_velocity, pipe_distance, top_pipe_height, bottom_pipe_height], dtype=np.float32)

    def one_hot(self, action_index):
        """
        Convert action index to one-hot encoding.
        
        Args:
            action_index: index of the action (0 for jump, 1 for do_nothing)
        Returns:
            one_hot_vector: one-hot encoded vector representing the action
        """
        # For Flappy Bird, we have 2 possible actions (jump or do_nothing)
        one_hot_vector = np.zeros(self.action_dim)
        one_hot_vector[action_index] = 1.0
        return one_hot_vector

    def REWARD(self, env_score, env_mileage, is_game_over, done_type=None):
        """
        Calculate reward based on game state.
        
        Args:
            env_score: number of pipes passed
            env_mileage: number of frames survived
            is_game_over: whether the game is over
            done_type: type of game over ('well_done' for success, None or other for failure)
            
        Returns:
            reward: numerical reward value
        """
        reward = 0.0
        
        # Calculate score delta (reward for passing pipes)
        score_delta = env_score - self.last_score
        if score_delta > 0:
            # Significant reward for passing a pipe
            reward += 10.0 * score_delta
        
        # Calculate mileage delta (reward for surviving)
        mileage_delta = env_mileage - self.last_mileage
        if mileage_delta > 0:
            # Small positive reward for each step survived
            reward += 0.1 * mileage_delta
        
        # Penalty for game over (unless it's a successful completion)
        if is_game_over:
            if done_type == 'well_done':
                # Big reward for successfully completing a level
                reward += 50.0
            else:
                # Negative reward for failing
                reward -= 15.0
        
        # Update the tracking variables
        self.last_score = env_score
        self.last_mileage = env_mileage
        
        return reward

    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        Use epsilon-greedy policy to choose an action.
        
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        # Convert the state dict to the input format expected by our model
        state_features = self.build_state(state)
        
        # Epsilon-greedy action selection
        if self.mode == 'train' and np.random.rand() < self.epsilon:
            # Exploration: choose random action
            action_idx = np.random.randint(0, 2)  # 0 for jump, 1 for do_nothing
        else:
            # Exploitation: choose best action according to the Q-network
            q_values = self.network.predict(state_features.reshape(1, -1))
            action_idx = np.argmax(q_values)
        
        # Store the current state for use in after_action_observation
        self.prev_state = state_features
        
        # Map the action index to the actual action code
        if action_idx == 0:
            self.prev_action = 0  # Store action index (0 = jump)
            return action_table['jump']
        else:
            self.prev_action = 1  # Store action index (1 = do_nothing)
            return action_table['do_nothing']

    def after_action_observation(self, state: dict, action_table: dict = None) -> None:
        """
        Process observations after an action has been taken and train the network.
        
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        if self.mode != 'train' or self.prev_state is None or self.prev_action is None:
            # Skip if not in training mode or no previous state-action pair
            return
        
        # Process next state
        next_state = self.build_state(state)
        
        # Calculate reward
        reward = self.REWARD(
            env_score=state.get('score', 0),
            env_mileage=state.get('mileage', 0),
            is_game_over=state.get('done', False),
            done_type=state.get('done_type', None)
        )
        
        # Store transition in replay memory
        self.storage.append((
            self.prev_state,         # s_t
            self.prev_action,        # a_t 
            reward,                  # r_t
            next_state,              # s_{t+1}
            state.get('done', False) # done flag
        ))
        
        # Sample and train only if we have enough samples
        if len(self.storage) >= self.n:
            # Sample minibatch from replay memory
            minibatch = random.sample(self.storage, self.n)
            
            # Unpack transitions
            states = np.vstack([exp[0] for exp in minibatch])
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch])
            next_states = np.vstack([exp[3] for exp in minibatch])
            dones = np.array([exp[4] for exp in minibatch], dtype=int)
            
            # Get Q values for next states using target network (Q_f)
            q_values_next = self.network2.predict(next_states)
            max_q_values_next = np.max(q_values_next, axis=1)
            
            # Calculate target values y_j
            targets = rewards + (1 - dones) * self.discount_factor * max_q_values_next
            
            # Get current Q values for states
            current_q_values = self.network.predict(states)
            
            # Create target array (a copy of current Q values)
            target_q_values = current_q_values.copy()
            
            # Update target for the selected actions
            for i in range(self.n):
                target_q_values[i, actions[i]] = targets[i]
            
            # Create mask for loss function (focus only on the actions taken)
            mask = np.zeros_like(target_q_values)
            for i in range(self.n):
                mask[i, actions[i]] = 1
            
            # Train the network
            self.network.fit_step(states, target_q_values, mask)
            
            # Increment training steps counter
            self.training_steps += 1
            
            # Update target network periodically
            if self.training_steps % self.update_target_freq == 0:
                MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        # Update previous state and action for next iteration
        if not state.get('done', False):
            self.prev_state = next_state
            # Infer the previous action from bird velocity if action_table is provided
            if action_table is not None:
                self.prev_action = 0 if state.get('bird_velocity', 0) < 0 else 1
        else:
            # Reset state and action tracking when game is over
            self.prev_state = None
            self.prev_action = None
            self.last_score = 0
            self.last_mileage = 0

    def receive_after_action_observation(self, state: dict, action_table: dict = None) -> None:
        """
        Process observations after an action has been taken.
        
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        self.after_action_observation(state, action_table)

    def train_network(self):
        """
        Train the Q-network using a batch of experiences from the replay memory.
        """
        # Sample a minibatch from replay memory
        minibatch = random.sample(self.storage, self.n)
        
        states = np.vstack([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.vstack([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch], dtype=int)
        
        # Calculate target Q-values
        q_values_next = self.network2.predict(next_states)
        max_q_values_next = np.max(q_values_next, axis=1)
        
        # Calculate target using Bellman equation
        targets = rewards + (1 - dones) * self.discount_factor * max_q_values_next
        
        # Get current Q values and create targets for all actions
        current_q_values = self.network.predict(states)
        target_q_values = current_q_values.copy()
        
        # Update only the Q values for the actions that were taken
        for i in range(self.n):
            target_q_values[i, actions[i]] = targets[i]
        
        # Create a mask for the loss function (only consider the actions that were taken)
        mask = np.zeros_like(target_q_values)
        for i in range(self.n):
            mask[i, actions[i]] = 1
        
        # Train the network
        self.network.fit_step(states, target_q_values, mask)
        
        # Increment step counter
        self.training_steps += 1
        
        # Update the target network periodically
        if self.training_steps % self.update_target_freq == 0:
            MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)
        # Also update the target network
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Change working directory to the script's directory
    config_path = os.path.join(script_dir, 'config.yml')

    parser = argparse.ArgumentParser(description='Run Flappy Bird with MyAgent.')
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path=config_path, show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    episodes = 10000
    
    # For tracking progress
    best_score = -float('inf')
    best_mileage = -float('inf')
    
    for episode in range(episodes):
        env.play(player=agent)

        # Log progress
        print(f"Episode {episode}, Score: {env.score}, Mileage: {env.mileage}, Epsilon: {agent.epsilon:.4f}")
        
        # Save the model if it's the best so far
        if env.score > best_score or (env.score == best_score and env.mileage > best_mileage):
            best_score = env.score
            best_mileage = env.mileage
            agent.save_model(path='my_model.ckpt')
            print(f"New best model saved! Score: {best_score}, Mileage: {best_mileage}")
        
        # Clear memory every 10 episodes to prevent obsolete q-values
        if episode % 10 == 0 and episode > 0:
            agent.storage.clear()
            print("Memory cleared!")

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    mileages = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)
        mileages.append(env2.mileage)
        print(f"Eval Episode {episode}, Score: {env2.score}, Mileage: {env2.mileage}")

    print(f"Max Score: {np.max(scores)}")
    print(f"Mean Score: {np.mean(scores)}")
    print(f"Max Mileage: {np.max(mileages)}")
    print(f"Mean Mileage: {np.mean(mileages)}")