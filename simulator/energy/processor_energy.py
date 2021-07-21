#!/usr/bin/env python3

from simulator.energy.core_energy import CoreEnergy


class ProcessorEnergy:

    def __init__(self, config, pt, hw, proc_id):
        self.config = config
        self.pt = pt
        self.hw = hw
        self.proc_id = proc_id

        # initialize energy items for this module
        self.energy_item = {}

        # sub-modules
        self.core_array = {}
        for i in range(config["num_core_x"]):
            for j in range(config["num_core_y"]):
                core_tr = self.pt[
                    "core_{}".format((i, j))
                ]
                core_instance = CoreEnergy(
                    config=config,
                    pt=core_tr,
                    hw=hw,
                    core_id=(i, j)
                )
                self.core_array[(i, j)] = core_instance

    def get_energy_metrics(self):
        """Gets a dictionary of energy metrics. """
        # Collect the energy metrics of this hardware module.
        energy_metrics = self.energy_item
        # evaluate core energy
        for i in range(self.config["num_core_x"]):
            for j in range(self.config["num_core_y"]):
                core_metrics = self.core_array[(i, j)].get_energy_metrics()
                assert len(core_metrics) == 1
                energy_metrics.update(core_metrics)
        
        return {"proc_{}".format(self.proc_id): energy_metrics}

