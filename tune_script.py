#! /usr/bin/env python

import socket

from ppo.main import exp_cli

if __name__ == "__main__":

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    exp_cli()
