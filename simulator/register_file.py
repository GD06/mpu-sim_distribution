import simpy 
from copy import deepcopy


class RegReadReq:

    def __init__(self, reg_addr):
        self.reg_addr = reg_addr
        return 


class RegReadResp:

    def __init__(self, reg_addr, data):
        self.reg_addr = reg_addr
        self.data = data
        return 


class RegWriteReq:

    def __init__(self, reg_addr, data):
        self.reg_addr = reg_addr
        self.data = data
        return


class RegWriteAck:

    def __init__(self, reg_addr):
        self.reg_addr = reg_addr 
        return 


class RegisterFile:

    def __init__(self, env, log, config, clock_unit, reg_file_type):
        """Multi-bank register file. Each bank has a separate read queue and 
        write queue of equal size. Each bank can serve a read request and 
        a write request at the same cycle.
        Args:
            env: simpy environment
            log: python log
            config: configuration dictionary
            clock_unit: register file clock unit
            reg_file_type: type of registrer file (near-bank or far-bank)
        """

        self.env = env 
        self.log = log
        self.config = config
        self.clock_unit = deepcopy(clock_unit)
        self.reg_file_type = reg_file_type
        
        self.reg_file_addr_map = self.config["reg_file_addr_map"]
        self.data_path_unit_size = self.config["data_path_unit_size"]
        self.num_threads_per_warp = self.config["num_threads_per_warp"]
        self.alignment = self.data_path_unit_size * self.num_threads_per_warp

        if self.reg_file_type == "far-bank":
            self.size = self.config["subcore_reg_file_size"] 
            self.array = bytearray(self.size)
            self.num_bank = self.config["num_subcore_reg_file_bank"]
            self.bank_queue_size = \
                self.config["subcore_reg_file_bank_queue_size"]
            self.read_latency = self.config["subcore_reg_file_read_latency"]
            self.write_latency = self.config["subcore_reg_file_write_latency"]
        elif self.reg_file_type == "near-bank":
            self.size = self.config["pg_reg_file_size"]
            self.array = bytearray(self.size)
            self.num_bank = self.config["num_pg_reg_file_bank"]
            self.bank_queue_size = \
                self.config["pg_reg_file_bank_queue_size"]
            self.read_latency = self.config["pg_reg_file_read_latency"]
            self.write_latency = self.config["pg_reg_file_write_latency"]
        else:
            raise NotImplementedError(
                "Unknown register file type:{}".format(self.reg_file_type)
            )

        # Initialize queues inside arbiter
        self.read_req_queue = []
        self.write_req_queue = []
        for i in range(self.num_bank):
            self.read_req_queue\
                .append(simpy.Store(env, capacity=self.bank_queue_size))
            self.write_req_queue\
                .append(simpy.Store(env, capacity=self.bank_queue_size))

        # Initialize output queues for operand collector
        self.read_resp_queue = simpy.FilterStore(env)
        self.write_resp_queue = simpy.FilterStore(env) 

        # Spawn a process for each register bank
        for bank_id in range(self.num_bank):
            self.env.process(self._serve_read_request(bank_id))
            self.env.process(self._serve_write_request(bank_id))

        # performance counter
        self.num_read = 0
        self.num_write = 0
        return 

    def get_perf_metrics(self):
        """Get a dictionary of performance metrics."""
        perf_metrics = {}
        perf_metrics["num_read"] = self.num_read
        perf_metrics["num_write"] = self.num_write
        if self.reg_file_type == "far-bank":
            return {"fb_reg_file": perf_metrics}
        else:
            return {"nb_reg_file": perf_metrics}

    def _align_down(self, reg_addr):
        aligned_reg_addr = reg_addr - (reg_addr % self.alignment)
        return aligned_reg_addr

    def _align_up(self, reg_addr):
        aligned_reg_addr = ((reg_addr - 1) // self.alignment + 1) \
            * self.alignment
        return aligned_reg_addr

    def _serve_write_request(self, bank_id):
        while True:
            req = yield self.write_req_queue[bank_id].get()
            assert isinstance(req, RegWriteReq)
            assert req.reg_addr % self.alignment == 0
            # serve write request
            assert isinstance(req.data, bytearray), "The write transaction"\
                "to the register file should contain bytearray as the data"
            # perform the actual write
            self.array[req.reg_addr: req.reg_addr + self.alignment] = \
                deepcopy(req.data)
            # compose write ack
            resp = RegWriteAck(
                reg_addr=req.reg_addr
            )
            yield self.env.timeout(self.write_latency * self.clock_unit)
            yield self.write_resp_queue.put(resp)
            # update performance counter
            self.num_write += 1
            
    def _serve_read_request(self, bank_id):
        while True:
            req = yield self.read_req_queue[bank_id].get()
            assert isinstance(req, RegReadReq)
            assert req.reg_addr % self.alignment == 0
            # serve read request
            reg_data = deepcopy(self.array[req.reg_addr: 
                                req.reg_addr + self.alignment])
            # compose read response
            resp = RegReadResp(
                reg_addr=req.reg_addr,
                data=reg_data
            )
            yield self.env.timeout(self.read_latency * self.clock_unit)
            yield self.read_resp_queue.put(resp)
            # update performance counter
            self.num_read += 1

    def calc_bank_index(self, reg_addr):
        """calculate the register file bank index given the register 
        file base address
        Args:
            reg_addr: base address into the register file 
                (aligned to register bank width)
        Return:
            bank_index: register file bank index
        """

        if self.reg_file_addr_map == "reg_file_addr_map_1":
            bank_index = int((reg_addr / self.alignment) % self.num_bank)
            return bank_index
        else:
            raise NotImplementedError(
                "Unknown type of register address mapping: {}"
                .format(self.reg_file_addr_map)
            )

