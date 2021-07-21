#!/usr/bin/env python3

import argparse 

import util.mtx_api as mtx_api


def main():

    parser = argparse.ArgumentParser(
        description="Generate the random graphs in the MTX file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    ) 

    parser.add_argument("num_nodes", type=int, help="The number of nodes")
    parser.add_argument("num_edges", type=int, help="The number of edges")
    parser.add_argument("output_file", help="The file path of output MTX file")

    parser.add_argument("--directed", action="store_true")

    args = parser.parse_args() 

    ptr_row, ptr_col, ptr_val = mtx_api.gen_random_graph(
        args.num_nodes, args.num_edges, directed=args.directed)

    symmetric = (args.directed is False) 

    mtx_api.dump_mtx_file(args.output_file, args.num_nodes, args.num_nodes, 
                          ptr_row, ptr_col, ptr_val, symmetric)


if __name__ == "__main__":
    main() 
