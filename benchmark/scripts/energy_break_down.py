#!/usr/bin/env python3 

import argparse 
import json 


def main():

    parser = argparse.ArgumentParser(
        description="Compute the power breakdown from the power trace file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    parser.add_argument("power_trace_file", help="The trace file containing "
                        "a detailed trace of energy results")

    args = parser.parse_args() 

    with open(args.power_trace_file, "r") as f:
        json_dict = json.load(f)
        power_dict = json_dict["hardware"]

    others_dict = {"icache_power", "fetch_decode_power", "scoreboard_power",
                   "reg_track_table_power", "lsu_power", "lsu_remote_power",
                   "lsu_extension_power", "smem_power", "wb_commit_power"}

    alu_dict = {"fb_alu_power", "nb_alu_power", "sfu_power"}

    rf_opc_dict = {"fb_opc_power", "nb_opc_power", "fb_rf_power", 
                   "nb_rf_power"}

    dram_dict = {"dram_read_power", "dram_write_power", "dram_act_power",
                 "dram_pre_power", "dram_refresh_power"}

    tsv_dict = {"subcore_pg_bus_power"}

    noc_dict = {"net_bus_power", "net_buffer_power", "net_switch_ctrl_power",
                "net_switch_xbar_power", "net_switch_output_power",
                "net_switch_output_ctrl_power", "net_switch_output_clk_power"}

    total_power = 0

    dram_power = 0 
    for k in dram_dict:
        dram_power += power_dict[k]
    total_power += dram_power 
    print("DRAM Power: {} W".format(dram_power))

    alu_power = 0
    for k in alu_dict:
        alu_power += power_dict[k]
    total_power += alu_power 
    print("ALU Power: {} W".format(alu_power))

    rf_opc_power = 0
    for k in rf_opc_dict:
        rf_opc_power += power_dict[k]
    total_power += rf_opc_power 
    print("OPC+RF Power: {} W".format(rf_opc_power))

    tsv_power = 0
    for k in tsv_dict:
        tsv_power += power_dict[k]
    total_power += tsv_power 
    print("TSV Power: {} W".format(tsv_power))

    noc_power = 0
    for k in noc_dict:
        noc_power += power_dict[k]
    total_power += noc_power 
    print("NoC Power: {} W".format(noc_power))

    others_power = 0
    for k in others_dict:
        others_power += power_dict[k]
    total_power += others_power 
    print("Others Power: {} W".format(others_power))

    print("Total accumulated power:", total_power)
    print("Total power from the file:", power_dict["total_power"])


if __name__ == "__main__":
    main() 
