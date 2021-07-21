from simulator.network_bus import NetworkBus


def config_2d_mesh(hardware):
    """
        Configure network connection according 2D mesh topology
    """
    config = hardware.config
    ch_idx = config["network_channel_index"]
    # configure x-dim link
    for p_id_y in range(config["num_processor_y"]):
        for c_id_y in range(config["num_core_y"]):
            for p_id_x in range(config["num_processor_x"]):
                for c_id_x in range(config["num_core_x"]):
                    if (
                        p_id_x == config["num_processor_x"] - 1
                        and c_id_x == config["num_core_x"] - 1
                    ):
                        # This is an edge node
                        continue
                    # get current router
                    cur_router = hardware\
                        .processor_array[(p_id_x, p_id_y)]\
                        .core_array[(c_id_x, c_id_y)]\
                        .niu.router
                    # get next router
                    next_c_id_x = (c_id_x + 1) % config["num_core_x"]
                    next_p_id_x = \
                        (c_id_x + 1) // config["num_core_x"] + p_id_x
                    assert next_p_id_x < config["num_processor_x"]
                    next_router = hardware\
                        .processor_array[(next_p_id_x, p_id_y)]\
                        .core_array[(next_c_id_x, c_id_y)]\
                        .niu.router
                    # configure bus
                    is_onchip = (c_id_x + 1 == config["num_core_x"])
                    bus = NetworkBus(
                        env=hardware.env,
                        log=hardware.log,
                        config=config,
                        is_onchip=is_onchip,
                        uplink_router=next_router,
                        downlink_router=cur_router,
                        bus_id=(
                            (p_id_x, p_id_y), (c_id_x, c_id_y),
                            (next_p_id_x, p_id_y), (next_c_id_x, c_id_y)
                        )
                    )
                    bus.config_queue(
                        uplink_in_chan=next_router
                        .in_packet_chan[ch_idx["W"]],
                        uplink_out_chan=next_router
                        .out_packet_chan[ch_idx["W"]],
                        downlink_in_chan=cur_router
                        .in_packet_chan[ch_idx["E"]],
                        downlink_out_chan=cur_router
                        .out_packet_chan[ch_idx["E"]]
                    )
                    hardware.bus_array[
                        (p_id_x, p_id_y), (c_id_x, c_id_y),
                        (next_p_id_x, p_id_y), (next_c_id_x, c_id_y)
                    ] = bus

    # configure y-dim link
    for p_id_x in range(config["num_processor_x"]):
        for c_id_x in range(config["num_core_x"]):
            for p_id_y in range(config["num_processor_y"]):
                for c_id_y in range(config["num_core_y"]):
                    if (
                        p_id_y == config["num_processor_y"] - 1
                        and c_id_y == config["num_core_y"] - 1
                    ):
                        # This is an edge node
                        continue
                    # get current router
                    cur_router = hardware\
                        .processor_array[(p_id_x, p_id_y)]\
                        .core_array[(c_id_x, c_id_y)]\
                        .niu.router
                    # get next router
                    next_c_id_y = (c_id_y + 1) % config["num_core_y"]
                    next_p_id_y = \
                        (c_id_y + 1) // config["num_core_y"] + p_id_y
                    assert next_p_id_y < config["num_processor_y"]
                    next_router = hardware\
                        .processor_array[(p_id_x, next_p_id_y)]\
                        .core_array[(c_id_x, next_c_id_y)]\
                        .niu.router
                    # configure bus
                    is_onchip = (c_id_y + 1 == config["num_core_y"])
                    bus = NetworkBus(
                        env=hardware.env,
                        log=hardware.log,
                        config=config,
                        is_onchip=is_onchip,
                        uplink_router=next_router,
                        downlink_router=cur_router,
                        bus_id=(
                            (p_id_x, p_id_y), (c_id_x, c_id_y),
                            (p_id_x, next_p_id_y), (c_id_x, next_c_id_y)
                        )
                    )
                    bus.config_queue(
                        uplink_in_chan=next_router
                        .in_packet_chan[ch_idx["S"]],
                        uplink_out_chan=next_router
                        .out_packet_chan[ch_idx["S"]],
                        downlink_in_chan=cur_router
                        .in_packet_chan[ch_idx["N"]],
                        downlink_out_chan=cur_router
                        .out_packet_chan[ch_idx["N"]]
                    )
                    hardware.bus_array[
                        (p_id_x, p_id_y), (c_id_x, c_id_y),
                        (p_id_x, next_p_id_y), (c_id_x, next_c_id_y)
                    ] = bus
