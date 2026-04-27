#!/usr/bin/env python3
"""
Plot possible Stableswap LP token price vs CRV oracle price.

This is the composition of the two existing scripts:

- analyze_bad_debt.py: per-position recovery rate (% of face) for each
  unprofitable position in the bad-debt CRV market, swept over CRV price.
  At each CRV price the still-active (uncapped solvency < 100%) positions
  span a range of recovery rates — the yellow envelope, shaded by debt
  density.
- lp_price_vs_vault_price.py: the CRV LlamaLend Recovery pool's
  arbitrage-driven LP price as a function of the vault token's market
  price expressed as a percentage of face value.

Composition: at each CRV price, map the envelope of recovery rates
through the pool's LP-price function to obtain a band of possible LP
prices. The yellow region is that band (with debt-density shading); the
black line is the LP price implied by the next-to-liquidate position.
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
from matplotlib.patches import Patch

from networks import NETWORK
import analyze_bad_debt as bd
import lp_price_vs_vault_price as lp


WAD = bd.WAD
NEWSPAPER_YELLOW = bd.NEWSPAPER_YELLOW
DUST_DEBT = 1000 * WAD


def main():
    boa.fork(NETWORK)
    boa.env.eoa = "0x0000000000000000000000000000000000000001"

    # --- Bad-debt market state ---------------------------------------------
    controller = boa.loads_abi(bd.CONTROLLER_ABI, name="Controller").at(
        bd.CONTROLLER)
    amm = boa.loads_abi(bd.AMM_ABI, name="AMM").at(controller.amm())

    A = amm.A()
    p_o_now = amm.price_oracle()
    base_price = amm.get_base_price()

    def p_oracle_up(n):
        return int(base_price * pow((A - 1) / A, n))

    print(f"A = {A}")
    print(f"price_oracle (CRV/crvUSD) = {p_o_now/WAD:.6f}")

    print("\nFetching users_to_liquidate ...")
    positions = controller.users_to_liquidate(0, 0)
    print(f"Got {len(positions)} positions")

    user_bands = []
    for pos in positions:
        addr, x_u, y_u, debt_u, _ = pos[0], pos[1], pos[2], pos[3], pos[4]
        ns = amm.read_user_tick_numbers(addr)
        n1, n2 = ns[0], ns[1]
        xs_ys = amm.get_xy(addr)
        xs = list(xs_ys[0])
        ys = list(xs_ys[1])
        bands = []
        for i, n in enumerate(range(n1, n2 + 1)):
            pu = p_oracle_up(n)
            y0 = bd.get_y0(xs[i], ys[i], p_o_now, pu, A)
            bands.append({"p_o_up": pu, "y0": y0})
        user_bands.append({"user": addr, "debt": debt_u, "bands": bands})

    # --- Pool state and LP-price lookup ------------------------------------
    pool = boa.loads_abi(lp.POOL_ABI, name="Pool").at(lp.POOL)
    vault = boa.loads_abi(lp.VAULT_ABI, name="Vault").at(lp.VAULT)
    pool_name = pool.name()
    A_pool = pool.A()
    bals = list(pool.get_balances())
    rates = list(pool.stored_rates())
    total_supply = pool.totalSupply()
    pps = vault.pricePerShare()

    rate1 = rates[1]
    xp = [bals[i] * rates[i] // WAD for i in range(2)]
    D = lp.get_D(xp, A_pool)
    peg = rate1  # crvUSD/share, ≈ pps/√2

    # Current implied vault price (% of face) and LP price.
    m_now = lp.marginal_price_scaled(xp[0], xp[1], D, A_pool)
    pct_now = (m_now * peg / pps) * 100.0
    lp_now = (xp[0] + xp[1] * m_now) / total_supply
    print(f"\npool {pool_name!r}")
    print(f"current implied vault price = {pct_now:.4f}% of face")
    print(f"current LP price            = {lp_now:.6f} crvUSD")

    # Lookup table: vault % of face -> LP price (crvUSD).
    pct_lookup = np.linspace(0.5, 105.0, 419)
    lp_lookup = np.empty_like(pct_lookup)
    for i, pct in enumerate(pct_lookup):
        market_price_int = int(pct / 100.0 * pps)
        m_target = market_price_int / peg
        x0 = lp.find_xp0_for_m(m_target, D, A_pool)
        x1 = lp.get_y(x0, D, A_pool)
        lp_lookup[i] = (x0 + x1 * m_target) / total_supply

    lp_full = float(np.interp(100.0, pct_lookup, lp_lookup))
    print(f"LP price at full recovery   = {lp_full:.6f} crvUSD")

    def lp_from_pct(pct_arr):
        """Map vault recovery rate (% of face) to LP price via lookup."""
        arr = np.asarray(pct_arr, dtype=float)
        out = np.full(arr.shape, np.nan)
        finite = ~np.isnan(arr)
        out[finite] = np.interp(arr[finite], pct_lookup, lp_lookup)
        return out

    # --- Sweep CRV price and compute per-user solvency ---------------------
    p_min = 0.01
    p_max = 1.3
    p_compute_max = 1.5
    n_pts = 460
    prices = np.linspace(p_min, p_compute_max, n_pts)

    nonzero = [u for u in user_bands if u["debt"] > 0]
    per_user_value = np.zeros((len(nonzero), len(prices)))
    for ui, u in enumerate(nonzero):
        for i, pf in enumerate(prices):
            p_wad = pf * WAD
            v = 0.0
            for b in u["bands"]:
                xb, yb = bd.xy_at_price(b["y0"], b["p_o_up"], p_wad, A)
                v += xb + p_wad * yb / WAD
            per_user_value[ui, i] = v

    debts = np.array([float(u["debt"]) for u in nonzero])
    per_user_solvency = per_user_value / debts[:, None] * 100.0

    big_mask = debts >= DUST_DEBT
    n_big = int(big_mask.sum())
    big_debts = debts[big_mask]
    big_solvency = per_user_solvency[big_mask]
    total_big = big_debts.sum()
    # Active = position hasn't yet been profitable to hard-liquidate.
    big_solvency_active = np.where(big_solvency < 100.0,
                                   big_solvency, np.nan)
    env_low = np.nanmin(big_solvency_active, axis=0)
    env_high = np.nanmax(big_solvency_active, axis=0)
    worst_remaining = np.full_like(prices, np.nan)
    for j in range(len(prices)):
        active = big_solvency[:, j] < 100.0
        if active.any():
            worst_remaining[j] = big_solvency[active, j].max()

    # Map envelope through LP-price lookup. Where the envelope has
    # collapsed (all positions liquidated → vault back at face), LP price
    # is just the full-recovery LP price.
    lp_env_low = lp_from_pct(env_low)
    lp_env_high = lp_from_pct(env_high)
    lp_worst = lp_from_pct(worst_remaining)
    lp_per_user = lp_from_pct(big_solvency_active)

    collapsed = np.isnan(env_low)
    lp_env_low[collapsed] = lp_full
    lp_env_high[collapsed] = lp_full
    lp_worst[collapsed] = lp_full

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.fill_between(prices, lp_env_low, lp_env_high,
                    color=NEWSPAPER_YELLOW, alpha=0.2, zorder=1, linewidth=0)
    # Debt-density shading: each position contributes one band from env_low
    # up to its own implied LP price, alpha ∝ debt share.
    ALPHA_SCALE = 1.5
    for i in range(n_big):
        a = ALPHA_SCALE * float(big_debts[i] / total_big)
        per_user_curve = lp_per_user[i].copy()
        # Where its solvency hit 100%, fill clip at lp_env_low (no
        # contribution past liquidation).
        per_user_curve = np.where(np.isnan(per_user_curve),
                                  lp_env_low, per_user_curve)
        ax.fill_between(prices, lp_env_low, per_user_curve,
                        color=NEWSPAPER_YELLOW, alpha=a, zorder=1,
                        linewidth=0)

    gradient_proxy = Patch(facecolor=NEWSPAPER_YELLOW, alpha=0.5,
                           label=f"liquidatable-debt density "
                                 f"(n={n_big} pos, debt > 1k crvUSD)")

    ax.plot(prices, lp_worst, color="black", linewidth=2.4, zorder=4,
            label="next-to-liquidate (top of still-remaining positions)")

    # Reference markers.
    ax.axhline(lp_full, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"full recovery LP price = {lp_full:.4f} crvUSD")
    ax.axhline(lp_now, color="darkorange", linestyle=":", linewidth=1.0,
               alpha=0.7,
               label=f"current LP price (implied) = {lp_now:.4f} crvUSD")
    ax.axvline(p_o_now / WAD, color="blue", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"current p_o = {p_o_now/WAD:.3f}")

    ax.set_xlabel("CRV price (crvUSD)")
    ax.set_ylabel("Stableswap LP token price (crvUSD)")
    ax.set_title(f"Possible Stableswap LP price vs CRV price  —  "
                 f"{pool_name}\nPool {lp.POOL}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(p_min, p_max)

    handles, labels = ax.get_legend_handles_labels()
    handles = [gradient_proxy] + handles
    labels = [gradient_proxy.get_label()] + labels
    ax.legend(handles, labels, loc="lower right", fontsize=10)

    out = "/home/michwill/Projects/stableswap-tools/plots/lp_price_vs_crv_price.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nSaved plot: {out}")
    plt.show()


if __name__ == "__main__":
    main()
