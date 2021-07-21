import simpy 
from copy import deepcopy 


class InstrLoadReq:

    def __init__(self, subcore_id, entry_id, pc):
        self.subcore_id = subcore_id
        self.entry_id = entry_id 
        self.pc = pc 


class InstrLoadResp:
    
    def __init__(self, subcore_id, entry_id, pc, instr):
        self.subcore_id = subcore_id
        self.entry_id = entry_id 
        self.pc = pc 
        self.instr = instr 


class InstrCache:

    def __init__(self, env, config, log, core):
        self.env = env 
        self.config = config 
        self.log = log 
        self.core = core 

        self.clock_unit = deepcopy(core.clock_unit) 

        total_num_warps = (
            config["num_subcore"] * config["max_num_warp_per_subcore"]
        )
        self.read_latency = self.config["icache_read_latency"]

        self.load_req_queque = simpy.Store(env, capacity=total_num_warps)
        self.load_resp_queue = simpy.FilterStore(env)

        self.num_port = self.config["num_port_icache"]
        # Spwan processes to serve load requests
        for i in range(self.num_port):
            self.env.process(self._req_handler())

        # performance metrics
        self.num_icache_read = 0

        return 

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics"""
        perf_metrics = {}
        perf_metrics["num_icache_read"] = self.num_icache_read
        return {"icache": perf_metrics}

    def _req_handler(self):
        while True:
            req = yield self.load_req_queque.get() 

            instr = self.core.current_kernel.instr_list[req.pc] 
            resp = InstrLoadResp(
                subcore_id=req.subcore_id, 
                entry_id=req.entry_id, 
                pc=req.pc, 
                instr=instr
            )

            yield self.env.timeout(self.read_latency * self.clock_unit)
            yield self.load_resp_queue.put(resp)

            # update performance counter
            self.num_icache_read += 1

        return 
