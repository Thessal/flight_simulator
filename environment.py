# https://pettingzoo.farama.org/content/environment_creation/
import airline

import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple, Sequence

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np

def env(cfg, render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(cfg, render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, cfg, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """      
        df_airline, df_preference, df_time = airline.get_df()
        self.airline_info = (df_airline, df_preference, df_time)
        self.possible_agents = [x for x in df_airline.values if (x.startswith("RK"))]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        # self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents} # policy(FIFO/Maxprofit/Urgentfirst)  
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents} # policy(FIFO/Maxprofit/Urgentfirst)  
        self._observation_spaces = {
            # agent: Sequence(Box(-60,60)) for agent in self.possible_agents # outgoing 10 airplanes delay
            agent: Tuple((Box(0,10,dtype=np.float32), Box(-60,60,dtype=np.float32))) for agent in self.possible_agents # outgoing airplane count, total delay
        }
        self.render_mode = render_mode
        self.num_iters = cfg["num_iters"]
        self.sim = airline.Simulator(cfg, dfs=self.airline_info) 

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return Sequence(Box(-60,60)) # time to planned departure
        return Tuple((Box(0,10,dtype=np.float32), Box(-60,60,dtype=np.float32))) # outgoing airplane count, total delay

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
        # return Tuple((Discrete(3),))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        string = "Current state:\n" + \
        "\n".join(f"{agent}:{airline.POLICY_NAMES[self.state[self.agents[0]]]}" for agent in self.agents)
        print(string)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self.observations[agent] #np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: np.array([airline.DEFAULT_POLICY], dtype=np.int64) for agent in self.agents}
        self.observations = {agent: (np.array([0.0], dtype=np.float32),np.array([0.0], dtype=np.float32)) for agent in self.agents}
        # self.state = {agent: airline.DEFAULT_POLICY for agent in self.agents}
        # self.observations = {agent: (0.0,0.0) for agent in self.agents}
        self.num_moves = 0
        self.sim.reset()       
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[agent] = action
        self.sim.airports[agent].policy = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
            #     (self.state[self.agents[0]], self.state[self.agents[1]])
            # ]
            rewards = self.sim.step()
            for agent0, agent1, reward0, reward1 in rewards:
                if (agent0 in self.rewards) :
                    self.rewards[agent0] += reward0
                if (agent1 in self.rewards):
                    self.rewards[agent1] += reward1

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= self.num_iters for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.sim.airports[i].observe()
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            # self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # for other_agent in self.agents:
            #     if agent!=other_agent:
            #         self.state[other_agent] = np.array([self.sim.airports[other_agent].policy], dtype=np.int64)
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()