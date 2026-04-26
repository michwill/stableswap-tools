#!/usr/bin/env python3
"""
Plot LP-token price for the "CRV LlamaLend Recovery" stableswap pool
(0x516c3ecfe45f0820653e08dd7c93633d71b93cb5) against the vault token's
market price expressed as a percentage of face value.

The pool holds (crvUSD, cvcrvUSD), where cvcrvUSD = ERC4626 lending vault
on the bad-debt CRV controller (0xEdA2…110A). Its rate oracle is a
PegPrice contract that returns a fixed value pricePerShare/√2 captured
at deploy — i.e. the pool prices a vault share at √2 less than face,
leaving room for the share to trade between ~70.7% and 100% of face
without arbitrage exhausting one side.

Math (Curve StableSwap, 2 coins, frictionless arbitrage):
- Scaled balances xp_i = b_i * rate_i / 1e18.
- Invariant D from Curve's iterative get_D.
- Pool's marginal price (asset 1 in asset 0, scaled units):
    m_scaled = (A·N·xp0 + Dr·xp0/xp1) / (A·N·xp0 + Dr)
  with Dr = D^(N+1) / (N^N · ∏xp_i), N=2.
- At external market price p (crvUSD per share), arbitrage drives
  m_scaled = p · 1e18 / rate1.
- LP price (crvUSD per LP) = (xp0 + xp1 · m_scaled) / total_supply.
"""

import os
import boa
import numpy as np
import matplotlib
if os.environ.get("MPL_NONINTERACTIVE"):
    matplotlib.use("Agg")
elif not os.environ.get("MPLBACKEND"):
    matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from networks import NETWORK
import analyze_bad_debt as bd


POOL = "0x516c3ecfe45f0820653e08dd7c93633d71b93cb5"
VAULT = "0xCeA18a8752bb7e7817F9AE7565328FE415C0f2cA"
DUST_DEBT = 1000 * 10**18  # ignore dust positions in the bad-debt scan
LOW_CRV_PRICE = 0.01 * 10**18  # CRV → 0 limit for "leftmost" envelope value

POOL_ABI = """[
    {"name":"A","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"name","outputs":[{"type":"string"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"get_balances","outputs":[{"type":"uint256[]"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"stored_rates","outputs":[{"type":"uint256[]"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"totalSupply","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"get_virtual_price","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"}
]"""

VAULT_ABI = """[
    {"name":"pricePerShare","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"}
]"""

WAD = 10**18
N = 2
A_PRECISION = 100


def get_D(xp, A_true):
    """Iterative D solver matching Curve StableSwap.get_D (integer math)."""
    amp = A_true * A_PRECISION
    S = sum(xp)
    if S == 0:
        return 0
    D = S
    Ann = amp * N
    for _ in range(255):
        D_P = D * D // xp[0] * D // xp[1] // (N**N)
        Dprev = D
        D = ((Ann * S // A_PRECISION + D_P * N) * D
             // ((Ann - A_PRECISION) * D // A_PRECISION + (N + 1) * D_P))
        if abs(D - Dprev) <= 1:
            return D
    raise RuntimeError("get_D didn't converge")


def get_y(x0, D, A_true):
    """Given xp[0]=x0 and invariant D, return xp[1]. Mirrors Curve's get_y."""
    amp = A_true * A_PRECISION
    Ann = amp * N
    c = D * D // (x0 * N)
    c = c * D * A_PRECISION // (Ann * N)
    b = x0 + D * A_PRECISION // Ann  # without -D term; b - D used below
    y = D
    for _ in range(255):
        y_prev = y
        y = (y * y + c) // (2 * y + b - D)
        if abs(y - y_prev) <= 1:
            return y
    raise RuntimeError("get_y didn't converge")


def marginal_price_scaled(xp0, xp1, D, A_true):
    """Pool marginal price (asset1 in asset0, scaled), as a float."""
    x0, x1, Df = float(xp0), float(xp1), float(D)
    Dr = Df / (N ** N)
    Dr = Dr * Df / x0
    Dr = Dr * Df / x1
    num = A_true * N * x0 + Dr * x0 / x1
    den = A_true * N * x0 + Dr
    return num / den


def find_xp0_for_m(m_target, D, A_true, x0_lo=None, x0_hi=None):
    """Bisect xp[0] so that pool's marginal price equals m_target (scaled)."""
    if x0_lo is None:
        x0_lo = D // 1_000_000
    if x0_hi is None:
        x0_hi = D - D // 1_000_000
    # m is monotonic in xp[0]: more xp[0] => higher m (asset1 expensive in asset0).
    for _ in range(100):
        mid = (x0_lo + x0_hi) // 2
        if mid <= 0 or mid >= D:
            break
        y = get_y(mid, D, A_true)
        m = marginal_price_scaled(mid, y, D, A_true)
        if m < m_target:
            x0_lo = mid
        else:
            x0_hi = mid
        if x0_hi - x0_lo <= 1:
            break
    return (x0_lo + x0_hi) // 2


def top_of_envelope_at_low_crv():
    """Highest per-position recovery rate (% of debt) across non-dust insolvent
    positions in the bad-debt CRV market when CRV → 0. This is the leftmost
    height of the yellow envelope in analyze_bad_debt.py."""
    controller = boa.loads_abi(bd.CONTROLLER_ABI, name="BadDebtCtl"
                               ).at(bd.CONTROLLER)
    amm = boa.loads_abi(bd.AMM_ABI, name="BadDebtAmm").at(controller.amm())
    A = amm.A()
    p_o_now = amm.price_oracle()
    base = amm.get_base_price()
    def p_o_up(n):
        return int(base * pow((A - 1) / A, n))
    positions = controller.users_to_liquidate(0, 0)
    best = 0.0
    for p in positions:
        addr, debt = p[0], p[3]
        if debt < DUST_DEBT:
            continue
        ns = amm.read_user_tick_numbers(addr)
        xs_ys = amm.get_xy(addr)
        xs, ys = list(xs_ys[0]), list(xs_ys[1])
        sx = sy = 0.0
        for i, n in enumerate(range(ns[0], ns[1] + 1)):
            pu = p_o_up(n)
            y0 = bd.get_y0(xs[i], ys[i], p_o_now, pu, A)
            xb, yb = bd.xy_at_price(y0, pu, LOW_CRV_PRICE, A)
            sx += xb; sy += yb
        value = sx + LOW_CRV_PRICE * sy / bd.WAD
        rec = value / debt * 100.0
        best = max(best, rec)
    return best


def main():
    boa.fork(NETWORK)
    boa.env.eoa = "0x0000000000000000000000000000000000000001"

    pool = boa.loads_abi(POOL_ABI, name="Pool").at(POOL)
    vault = boa.loads_abi(VAULT_ABI, name="Vault").at(VAULT)

    pool_name = pool.name()
    A = pool.A()
    bals = list(pool.get_balances())
    rates = list(pool.stored_rates())
    total_supply = pool.totalSupply()
    pps = vault.pricePerShare()  # face value, 1e18-scaled crvUSD per share

    rate0, rate1 = rates[0], rates[1]
    xp = [bals[i] * rates[i] // WAD for i in range(2)]
    D = get_D(xp, A)

    peg = rate1  # crvUSD per share, 1e18-scaled (≈ pricePerShare/√2)

    print(f"pool          = {POOL}  ({pool_name!r})")
    print(f"A             = {A}")
    print(f"balances      = {bals[0]/WAD:>14,.4f} crvUSD , "
          f"{bals[1]/WAD:>14,.4f} cvcrvUSD")
    print(f"stored_rates  = {rate0/WAD:.6f} , {rate1/WAD:.10f}  (peg = "
          f"{peg/WAD:.10f})")
    print(f"xp (scaled)   = {xp[0]/WAD:>14,.4f} , {xp[1]/WAD:>14,.4f}")
    print(f"D             = {D/WAD:,.6f}")
    print(f"total_supply  = {total_supply/WAD:,.6f}")
    print(f"virtual_price = {D/total_supply:.6f}")
    print(f"vault PPS     = {pps/WAD:.10f}  (face value, ratio face/peg = "
          f"{pps/peg:.6f})")

    # Sanity check: at current xp the implied m_scaled should equal the
    # current implied scaled-price of asset 1 relative to peg in the pool.
    m_now = marginal_price_scaled(xp[0], xp[1], D, A)
    print(f"\ncurrent m_scaled (pool)        = {m_now:.6f}")
    print(f"current implied market price   = "
          f"{m_now * peg / WAD:.10f} crvUSD/share")
    print(f"current implied market % face  = "
          f"{(m_now * peg / pps) * 100:.4f}%")

    # Sweep market price as fraction of face.
    fractions = np.linspace(0.50, 1.00, 300)
    lp_prices = np.zeros_like(fractions)
    for i, frac in enumerate(fractions):
        market_price_int = int(frac * pps)            # crvUSD/share, 1e18
        m_target = market_price_int / peg              # scaled, dimensionless
        x0 = find_xp0_for_m(m_target, D, A)
        x1 = get_y(x0, D, A)
        # LP value in 1e18-scaled crvUSD:
        #   LP_val = b0 + b1 * market_price_per_share_crvUSD
        #         = xp0 + xp1 * (market_price/peg) [via b_i = xp_i / rate_i * 1e18]
        lp_value = x0 + x1 * m_target
        lp_prices[i] = lp_value / total_supply  # crvUSD per LP token

    # Plot
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    pct_axis = fractions * 100.0
    ax.plot(pct_axis, lp_prices, color="black", linewidth=2.4)

    pct_now = (m_now * peg / pps) * 100.0
    pct_top_envelope = top_of_envelope_at_low_crv()
    pct_full = 100.0
    print(f"\ntop of bad-debt envelope at CRV→0 = {pct_top_envelope:.4f}% of "
          f"face")

    lp_at_now = float(np.interp(pct_now, pct_axis, lp_prices))
    lp_at_top = float(np.interp(pct_top_envelope, pct_axis, lp_prices))
    lp_at_full = float(np.interp(pct_full, pct_axis, lp_prices))

    ax.set_xlim(50, 100)
    xleft = ax.get_xlim()[0]
    ymin = lp_prices.min()

    def mark(x, y, color, ls, label, ytext_offset=3):
        ax.vlines(x, ymin, y, color=color, linestyle=ls,
                  linewidth=1.0, alpha=0.8, label=label)
        ax.hlines(y, xleft, x, color=color, linestyle=ls,
                  linewidth=1.0, alpha=0.8)
        ax.annotate(f"{y:.4f}", xy=(xleft, y),
                    xytext=(4, ytext_offset), textcoords="offset points",
                    fontsize=9, color=color)

    mark(pct_top_envelope, lp_at_top, "darkorange", "--",
         f"top of bad-debt envelope at CRV→0 "
         f"({pct_top_envelope:.2f}% of face)", ytext_offset=-12)
    mark(pct_now, lp_at_now, "blue", ":",
         f"current implied market price ({pct_now:.2f}% of face)")
    mark(pct_full, lp_at_full, "red", "--",
         f"full recovery (vault token at 100% of face)")

    ax.set_xlabel("Vault token market price  (% of face value)")
    ax.set_ylabel("LP token price  (crvUSD)")
    ax.set_title(f"LP price vs vault-token discount  —  {pool_name}\n"
                 f"Pool {POOL}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    out = "/home/michwill/Projects/stableswap-tools/plots/lp_price_vs_vault_price.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nSaved plot: {out}")
    plt.show()


if __name__ == "__main__":
    main()
