import simpy
from copy import deepcopy


class ExecutionUnit:
    
    def __init__(self, env, log, config, clock_unit):
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = deepcopy(clock_unit)
        self.num_threads_per_warp = self.config["num_threads_per_warp"]

        self.instr_entry_queue = simpy.Store(env, capacity=1)
