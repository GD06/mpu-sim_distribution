from copy import deepcopy 

from simulator.core import Core 


class Processor:

    def __init__(self, proc_id, env, config, log, hardware):
        self.proc_id = proc_id 
        self.env = env
        self.config = config 
        self.log = log 
        self.hardware = hardware

        self.filter_func = hardware.filter_func 
        self.traceEvents = []

        self.core_array = {}
        for i in range(config["num_core_x"]):
            for j in range(config["num_core_y"]):
                core_instance = Core(
                    core_id=(i, j), 
                    env=env,
                    config=config,
                    log=log,
                    processor=self, 
                )
                self.core_array[(i, j)] = core_instance

        return

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        
        # Collect the performance metrics of all hardware sub-modules. 
        for core_id in self.core_array.keys():
            core_metrics = self.core_array[core_id].get_perf_metrics() 
            assert len(core_metrics) == 1
            perf_metrics.update(core_metrics) 

        return {"proc_{}".format(self.proc_id): perf_metrics}

    def get_trace_events(self):
        """Get a list of trace events"""
        # Collect the trace evenets of this hardware module 
        _traceEvenets = deepcopy(self.traceEvents)

        # Collect the trace evenets of all hardware sub-modules 
        for core_id in self.core_array.keys():
            _traceEvenets += self.core_array[core_id].get_trace_events() 
        return _traceEvenets 
