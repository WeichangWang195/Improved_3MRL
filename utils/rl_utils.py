from collections import namedtuple

# Transition = namedtuple('Transition', ['state', 'obs', 'prob', 'v_eval', 'action', 'r_ex',
#                                        'extrinsic_critic_advantage', 'baseline_value'])

# TabularQTransition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
#
# QTransition = namedtuple('Transition', ['q_input', 'ir_input', 'action', 'r_ex',
#                                         'next_q_input', 'next_ir_input', 'baseline_value'])
#
# QvalueTransition = namedtuple('Transition', ['Q_value', 'r_in', 'r_ex', 'next_state', 'next_Q_value', 'baseline_value'])
#
# EnvCrticRecord = namedtuple('SavedValueRecord', ['state', 'reward', 'next_state'])
#
# HLTransition = namedtuple('Transition', ['commander_obs', 'decoder_obs', 'command', 'action', 'reward',
#                                          'next_commander_obs', 'next_decoder_obs', 'baseline_value'])

Transition = namedtuple('Transition', ['state_input', 'actor_obs_list', 'critic_obs_list', 'neighbors', 'action',
                                       'reward', 'next_state_input', 'next_actor_obs_list', 'next_critic_obs_list',
                                       'baseline_value'])


class delay_buffer(object):
    def __init__(self, size):
        self.size = size
        self.queue = []

    def is_empty(self):
        return self.queue == []

    def is_full(self):
        return len(self.queue) >= self.size

    def push(self, x):
        if len(self.queue) >= self.size:
            self.queue.pop()
            self.queue.insert(0, x)
        else:
            self.queue.insert(0, x)

    def get(self, delay):
        if not self.queue:
            raise Exception('Buffer is Empty')
        elif delay >= self.size:
            raise Exception('Exceeds Delay Buffer')
        elif len(self.queue) > delay:
            return self.queue[delay]
        else:
            return self.queue[-1]

    def get_buffer(self):
        return self.queue

    def clear_buffer(self):
        self.queue = []





