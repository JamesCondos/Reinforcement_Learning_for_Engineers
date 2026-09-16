import numpy as np
import random
from collections import defaultdict

class QLAgent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        alpha=0.5,
        gamma=0.95,
        sarsa_lambda=0.0,
        look_ahead_step=0,
        epsilon=0.5,
        epsilon_decay=0.99,
        min_epsilon=0.01,
        init_qtable_value=0.0,
        off_policy = False, 
    ):
        self.off_policy = off_policy
        self.alpha = alpha
        self.gamma = gamma
        self.sarsa_lambda = sarsa_lambda
        self.look_ahead_step = look_ahead_step
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        if not off_policy and sarsa_lambda > 0:
            assert (self.look_ahead_step == 0), "SARSA(n) OR SARSA(lambda)"



        ### Q Table - called in the form of self.q_table[state][action]
        self.q_table = defaultdict(lambda: np.full(self.action_dim, init_qtable_value))  

        
        ### Eligible trace for SARSA(lambda)
        self.eligibility_trace = defaultdict(lambda: np.zeros(self.action_dim))

        
        ### Buffer for SARSA(n)
        if self.look_ahead_step > 0:
            self.nstep_buffer = []  # [(state, action, reward)]

    def sample_action(self, state, greedy=False):

        # Select an action using the epsilon-greedy strategy.
        if not greedy and random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            action = int(np.argmax(self.q_table[state]))
        
        return action
    
    
        
    def update(self, state, action, reward, next_state, next_action, done):
        
        ### Q Learning
        if self.off_policy:

            # Bootstrap from the greedy action in the next state.
            next_best_action = int(np.argmax(self.q_table[next_state]))

            td_target = reward + self.gamma * self.q_table[next_state][next_best_action] * (not done)

            td_error = td_target - self.q_table[state][action]
            
            self.q_table[state][action] += self.alpha * td_error


            if done:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)



        ### SARSA(lambda) and SARSA(0)
        elif self.sarsa_lambda >= 0 and self.look_ahead_step == 0:  
                        
            # Bootstrap from the action actually selected by the policy.
            td_target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
            
            td_error = td_target - self.q_table[state][action]

            # Accumulating eligibility traces.
            self.eligibility_trace[state][action] += 1

            for s in list(self.q_table):
                self.q_table[s] += self.alpha * td_error * self.eligibility_trace[s]
                self.eligibility_trace[s] *= self.gamma * self.sarsa_lambda
                

            if done:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                self.eligibility_trace.clear()
        


        ### SARSA(n)
        elif self.look_ahead_step > 0:
            self.nstep_buffer.append((state, action, reward))
            if len(self.nstep_buffer) < self.look_ahead_step and not done:
                return {"epsilon": self.epsilon, "td_error": 0,} 
            
            if not done:                
                n = min(self.look_ahead_step, len(self.nstep_buffer))

                # Add n rewards and bootstrap from the selected action.
                G = sum(
                    (self.gamma ** i) * transition[2]
                    for i, transition in enumerate(self.nstep_buffer[:n])
                )
                G += (self.gamma ** n) * self.q_table[next_state][next_action]

                ### Recall the state-action pair to update
                update_state, update_action, _ = self.nstep_buffer.pop(0)

                td_error = G - self.q_table[update_state][update_action]

                self.q_table[update_state][update_action] += self.alpha * td_error
                
                
            ### If the episode ended
            else:
                while self.nstep_buffer:
                    n = len(self.nstep_buffer)
                    
                    # No bootstrap is used after a terminal transition.
                    G = sum(
                        (self.gamma ** i) * transition[2]
                        for i, transition in enumerate(self.nstep_buffer)
                    )
                    
                    ### Recall the state-action pair to update
                    update_state, update_action, _ = self.nstep_buffer.pop(0)
                    
                    td_error = G - self.q_table[update_state][update_action]
                    
                    self.q_table[update_state][update_action] += self.alpha * td_error

                self.nstep_buffer.clear()

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return {
            "epsilon": self.epsilon,
            "td_error": td_error,
        }
