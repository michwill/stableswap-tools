#!/usr/bin/env python3

import boa
import os
import json
from math import sqrt
from time import sleep
from eth_account import account
from getpass import getpass
from boa.explorer import Etherscan
from boa.verifiers import verify as boa_verify

from networks import NETWORK
from networks import ETHERSCAN_API_KEY


FORK = False
EXTRA_TIMEOUT = 10
DEPLOYER = "0xbabe61887f1de2713c6f97e567623453d3C79f67"  # babe
PEG_PRICE = int(1250530656965887 / sqrt(2))  # pricePerShare of 0xCeA18a8752bb7e7817F9AE7565328FE415C0f2cA / sqrt(2)


def account_load(fname):
    path = os.path.expanduser(os.path.join('~', '.brownie', 'accounts', fname + '.json'))
    with open(path, 'r') as f:
        pkey = account.decode_keyfile_json(json.load(f), getpass())
        return account.Account.from_key(pkey)


def verify(*args, **kw):
    while True:
        try:
            sleep(EXTRA_TIMEOUT)
            boa_verify(*args, **kw)
            break
        except ValueError as e:
            print(e)
            if "Already Verified" in str(e):
                return


if __name__ == '__main__':
    if FORK:
        boa.fork(NETWORK)
    else:
        boa.set_network_env(NETWORK)
        etherscan = Etherscan(api_key=ETHERSCAN_API_KEY)

    if FORK:
        boa.env.eoa = DEPLOYER
    else:
        admin = account_load('babe')
        boa.env.add_account(admin)

    peg_price = boa.load('contracts/PegPrice.vy', PEG_PRICE)
    if not FORK:
        verify(peg_price, etherscan, wait=True)

    print(peg_price.address)
