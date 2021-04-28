from abc import ABCMeta, abstractmethod

class BaseAgent:
    """
    Defines the interface of an RLGlue Agent

    ie. These methods must be defined in your own Agent classes
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, idx, cfg):
        """Declare agent variables."""
        pass

    @abstractmethod
    def agent_init(self):
        """Initialize agent variables."""

    @abstractmethod
    def agent_start(self, state, state_input, actor_obs_list, eps, goal):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, state_new, command_new, decoder_obs_new):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

    @abstractmethod
    def agent_update(self, update_policy):
        """

        :param update_policy:
        :return:
        """

    @abstractmethod
    def agent_save(self, idx):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

    @abstractmethod
    def agent_message(self, message):
        """
        receive a message from rlglue
        args:
            message (str): the message passed
        returns:
            str : the agent's response to the message (optional)
        """


class BaseEnvironment:
    """
    Defines the interface of an RLGlue environment

    ie. These methods must be defined in your own environment classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, cfg):
        """Declare environment variables."""

    @abstractmethod
    def env_init(self, maze):
        """
        Initialize environment variables.
        """

    @abstractmethod
    def env_start(self, maze, start, goal, goal_idx):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    @abstractmethod
    def env_message(self, message):
        """
        receive a message from RLGlue
        Args:
           message (str): the message passed
        Returns:
           str: the environment's response to the message (optional)
        """