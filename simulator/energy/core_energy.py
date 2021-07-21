#!/usr/bin/env python3

from simulator.energy.pg_energy import ProcessingGroupEnergy
from simulator.energy.subcore_energy import SubcoreEnergy


class CoreEnergy:

    def __init__(self, config, pt, hw, core_id):
        self.config = config
        self.pt = pt
        self.hw = hw
        self.core_id = core_id

        # initialize energy items for this module
        self.energy_item = {}
        # remote load-store unit energy
        self.energy_item["lsu_remote_energy"] = 0
        # TSV energy
        self.energy_item["subcore_pg_bus_energy"] = 0
        # instruction cache energy
        self.energy_item["icache_energy"] = 0
        # shared memory energy
        self.energy_item["smem_energy"] = 0

        # sub-modules
        self.subcore_array = []
        for i in range(config["num_subcore"]):
            subcore_tr = self.pt[
                "subcore_{}".format(i)
            ]
            subcore = SubcoreEnergy(
                config=config,
                pt=subcore_tr,
                hw=hw,
                subcore_id=i
            )
            self.subcore_array.append(subcore)

        self.pg_array = []
        for i in range(config["num_pg"]):
            pg_tr = self.pt[
                "pg_{}".format(i)
            ]
            pg = ProcessingGroupEnergy(
                config=config,
                pt=pg_tr,
                hw=hw,
                pg_id=i
            )
            self.pg_array.append(pg)

    def _update_lsu_remote_energy(self):
        tr = self.pt["lsu_remote"]
        self.energy_item["lsu_remote_energy"] = \
            tr["num_prt_read"] * self.config["lsu_remote_prt_read_energy"] \
            + tr["num_prt_write"] * self.config["lsu_remote_prt_write_energy"]
        self.hw.energy_item["lsu_remote_energy"] += \
            self.energy_item["lsu_remote_energy"]

    def _update_icache_energy(self):
        tr = self.pt["icache"]
        self.energy_item["icache_energy"] += tr["num_icache_read"] \
            * self.config["icache_read_energy"]
        self.hw.energy_item["icache_energy"] += \
            self.energy_item["icache_energy"]

    def _update_subcore_pg_bus_energy(self):
        tr = self.pt["subcore_pg_bus"]
        self.energy_item["subcore_pg_bus_energy"] += tr["num_bus_cyc"] \
            * (self.config["core_shared_bus_io_width"] * 8) \
            * self.config["core_shared_bus_energy_bit"]
        self.hw.energy_item["subcore_pg_bus_energy"] += \
            self.energy_item["subcore_pg_bus_energy"]

    def _update_smem_energy(self):
        tr = self.pt["smem"]
        self.energy_item["smem_energy"] += \
            tr["num_smem_bank_read"] * self.config["smem_bank_read_energy"] \
            + tr["num_smem_bank_write"] * self.config["smem_bank_write_energy"]
        self.hw.energy_item["smem_energy"] += self.energy_item["smem_energy"]

    def get_energy_metrics(self):
        """Gets a dictionary of energy metrics. """
        # Collect the energy metrics of this hardware module.
        self._update_lsu_remote_energy()
        self._update_icache_energy()
        self._update_subcore_pg_bus_energy()
        self._update_smem_energy()
        energy_metrics = self.energy_item
        # update subcore energy
        for i in range(self.config["num_subcore"]):
            subcore_metrics = self.subcore_array[i].get_energy_metrics()
            assert len(subcore_metrics) == 1
            energy_metrics.update(subcore_metrics)
        # update processing group energy
        for i in range(self.config["num_pg"]):
            pg_metrics = self.pg_array[i].get_energy_metrics()
            assert len(pg_metrics) == 1
            energy_metrics.update(pg_metrics)

        return {"core_{}".format(self.core_id): energy_metrics}
