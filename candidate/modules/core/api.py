import argparse
from importlib import import_module
import os
import sys


def check_main(name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-module", type=str)
    args, unknow = parser.parse_known_args()
    return args.api_module == name


def callmodule(module, args):
    sys.argv = [f"{module}.py", "--api-module", module] + args
    module = import_module(module)
    module.api = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True)
    args, unknow = parser.parse_known_args()

    callmodule(args.module, unknow)
