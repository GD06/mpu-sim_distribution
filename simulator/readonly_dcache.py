from copy import deepcopy
import random


class ReadOnlyDataCache:

    def __init__(self, name, env, log, config, clock_unit, granularity):
        self.name = name
        self.env = env
        self.log = log
        self.config = config
        self.clock_unit = clock_unit
        self.size = self.config[name + "_readonly_dcache_size"]
        assert self.size % granularity == 0
        self.num_entry = self.size // granularity
        self.latency = self.config[name + "_readonly_dcache_latency"]
        self.eviction_policy = \
            self.config[name + "_readonly_dcache_eviction_policy"]
        self.dcache = {}
        self.addr_time = {}

    def _cache_evict(self):
        if self.eviction_policy == "lru":
            addr_to_evict = [
                mem_addr for mem_addr in self.addr_time
                if all(
                    self.addr_time[tmp_mem_addr] >= self.addr_time[mem_addr] 
                    for tmp_mem_addr in self.addr_time
                )
            ]
            if len(addr_to_evict) >= 1:
                del self.addr_time[addr_to_evict[0]]
                del self.dcache[addr_to_evict[0]]
            else:
                assert False
        elif self.eviction_policy == "random":
            addr_to_evict = random.choice(tuple(self.addr_time.keys()))
            del self.addr_time[addr_to_evict]
            del self.dcache[addr_to_evict]
        else:
            assert False, "wrong eviction policy: {}"\
                .format(self.eviction_policy)

    def update(self, mem_addr, data):
        assert isinstance(data, bytearray)
        if mem_addr in self.dcache:
            assert self.dcache[mem_addr] == data
            self.addr_time[mem_addr] = self.env.now
        elif len(self.dcache) == self.num_entry:
            # data cache is full
            self._cache_evict()
            self.dcache[mem_addr] = deepcopy(data)
            self.addr_time[mem_addr] = self.env.now
        elif len(self.dcache) < self.num_entry:
            self.dcache[mem_addr] = deepcopy(data)
            self.addr_time[mem_addr] = self.env.now
        else:
            assert False

    def read(self, mem_addr):
        yield self.env.timeout(self.latency * self.clock_unit)
        if mem_addr in self.dcache:
            self.addr_time[mem_addr] = self.env.now
            return True, deepcopy(self.dcache[mem_addr])
        else:
            return False, None

