#!/usr/bin/env python3

import sys 

max_power = 0

try:
    for line in sys.stdin:
        values = line.split(" ")
        curr_power = float(values[0])
        max_power = max(max_power, curr_power)
        print("Max power: {:.2f} W".format(max_power), end="\r")
except KeyboardInterrupt: 
    pass 

print("")
