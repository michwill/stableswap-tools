# pragma version ~=0.4.3

peg_price: public(immutable(uint256))


@deploy
def __init__(_peg_price: uint256):
    peg_price = _peg_price
