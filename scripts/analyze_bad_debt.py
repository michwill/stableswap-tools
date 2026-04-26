#!/usr/bin/env python3
"""
Analyze unprofitable-to-liquidate positions in the CRV crvUSD market.

For each user returned by Controller.users_to_liquidate(), compute what amounts
of crvUSD (x) and CRV (y) they would have in the AMM if the CRV oracle price
were p, then plot recovery = (x + p*y - debt)/debt * 100% vs p.

Math (LLAMMA, per band):
    y0 = invariant amount of collateral if x=0 and p_band == p_o == p_o_up
    For new oracle price p (adiabatic, no fees):
      p >= p_o_up(n)        : y_band = y0,                x_band = 0
      p <= p_o_down(n)      : y_band = 0,                 x_band = y0 * p_o_down
      p_o_down < p < p_o_up : y_band = A*y0*(p-p_o_down)/p,
                              x_band = (A-1)*y0*p_o_up + p*y_band - A*y0*p^2/p_o_up
    y0 scales linearly with user share, so per-user y0 can be computed directly
    from user's per-band (x_user, y_user) via the standard _get_y0 formula.

Reads chain via titanoboa.
"""

import os
import boa
import numpy as np
import matplotlib
# Default: interactive UI via QtAgg (PyQt6 is in deps). Override via MPLBACKEND.
# Set MPL_NONINTERACTIVE=1 to use Agg (no window, save-only).
if os.environ.get("MPL_NONINTERACTIVE"):
    matplotlib.use("Agg")
elif not os.environ.get("MPLBACKEND"):
    matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from networks import NETWORK


CONTROLLER = "0xEdA215b7666936DEd834f76f3fBC6F323295110A"

CONTROLLER_ABI = """[
    {"name":"amm","outputs":[{"type":"address"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"users_to_liquidate","outputs":[{"type":"tuple[]","components":[
        {"name":"user","type":"address"},
        {"name":"x","type":"uint256"},
        {"name":"y","type":"uint256"},
        {"name":"debt","type":"uint256"},
        {"name":"health","type":"int256"}]}],
     "inputs":[{"name":"_from","type":"uint256"},{"name":"_limit","type":"uint256"}],
     "stateMutability":"view","type":"function"},
    {"name":"debt","outputs":[{"type":"uint256"}],
     "inputs":[{"name":"_user","type":"address"}],"stateMutability":"view","type":"function"},
    {"name":"user_state","outputs":[{"type":"uint256[4]"}],
     "inputs":[{"name":"_user","type":"address"}],"stateMutability":"view","type":"function"},
    {"name":"total_debt","outputs":[{"type":"uint256"}],
     "inputs":[],"stateMutability":"view","type":"function"}
]"""

AMM_ABI = """[
    {"name":"A","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"price_oracle","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"active_band","outputs":[{"type":"int256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"get_p","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"get_base_price","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
    {"name":"read_user_tick_numbers","outputs":[{"type":"int256[2]"}],
     "inputs":[{"name":"user","type":"address"}],"stateMutability":"view","type":"function"},
    {"name":"get_xy","outputs":[{"type":"uint256[][2]"}],
     "inputs":[{"name":"user","type":"address"}],"stateMutability":"view","type":"function"}
]"""


WAD = 10**18

# Aged-newsprint pale yellow for the debt-density shading.
NEWSPAPER_YELLOW = "#d4b970"


def get_y0(x, y, p_o, p_o_up, A):
    """Compute band invariant y0.

    For unsaturated bands (both x>0 and y>0), y0 is recovered from the
    AMM-style quadratic at the current oracle price. For saturated bands,
    the stored (x, y) reflects the boundary state from when the AMM last
    crossed the band edge, and y0 is read directly from that boundary
    condition (independent of current oracle price):
      x>0, y=0  ->  band saturated at lower edge: x = y0 * p_o_down  =>  y0 = x*WAD/p_o_down
      x=0, y>0  ->  band saturated at upper edge: y = y0             =>  y0 = y
    """
    if x == 0 and y == 0:
        return 0
    if y == 0:
        p_o_down = p_o_up * (A - 1) // A
        return x * WAD // p_o_down
    if x == 0:
        return y
    if p_o == 0:
        return 0
    Aminus1 = A - 1
    b = p_o_up * Aminus1 * x // p_o
    b += A * p_o**2 // p_o_up * y // WAD
    D = b * b + (4 * A * p_o) * y // WAD * x
    return (b + int_isqrt(D)) * WAD // (2 * A * p_o)


def isqrt(n):
    return int(np.floor(np.sqrt(float(n)))) if n < 2**53 else int_isqrt(n)


def int_isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def xy_at_price(y0, p_o_up, p, A):
    """Adiabatic (x, y) for a band given band invariant y0 and new oracle price p.

    All prices (p, p_o_up) are scaled by 1e18; y0, x, y are token wei (1e18 for
    18-decimal tokens). The /WAD factors below convert price*token products
    back to token wei.
    """
    if y0 == 0:
        return 0.0, 0.0
    p_o_down = p_o_up * (A - 1) / A
    if p >= p_o_up:
        return 0.0, y0
    if p <= p_o_down:
        return y0 * p_o_down / WAD, 0.0
    y = A * y0 * (p - p_o_down) / p
    x = ((A - 1) * y0 * p_o_up
         + p * y
         - A * y0 * p * p / p_o_up) / WAD
    if x < 0:
        x = 0.0
    return x, y


def main():
    # Fork mode for cheap read-only access; no EOA required.
    boa.fork(NETWORK)
    boa.env.eoa = "0x0000000000000000000000000000000000000001"

    controller = boa.loads_abi(CONTROLLER_ABI, name="Controller").at(CONTROLLER)
    amm_addr = controller.amm()
    print(f"AMM: {amm_addr}")
    amm = boa.loads_abi(AMM_ABI, name="AMM").at(amm_addr)

    A = amm.A()
    p_o_now = amm.price_oracle()
    p_amm = amm.get_p()
    active = amm.active_band()
    base_price = amm.get_base_price()
    print(f"A = {A}")
    print(f"price_oracle (CRV/crvUSD, 1e18) = {p_o_now}  -> {p_o_now / WAD:.6f}")
    print(f"AMM p (1e18)                     = {p_amm}  -> {p_amm / WAD:.6f}")
    print(f"active_band = {active}")
    print(f"base_price = {base_price}  -> {base_price / WAD:.6f}")

    # Local computation of p_oracle_up(n) = base_price * ((A-1)/A) ** n.
    # AMM does this with wad_exp; for our purposes plain float power is fine.
    def p_oracle_up_local(n):
        return int(base_price * pow((A - 1) / A, n))

    market_debt = controller.total_debt()
    print(f"market total_debt = {market_debt/WAD:,.2f} crvUSD")

    print("\nFetching users_to_liquidate ...")
    positions = controller.users_to_liquidate(0, 0)
    print(f"Got {len(positions)} positions")

    users = []
    for p in positions:
        # boa returns named-tuple-ish; index access works
        users.append({
            "user": p[0],
            "x": p[1],
            "y": p[2],
            "debt": p[3],
            "health": p[4],
        })

    # For each user, gather per-band (x_user, y_user) and p_o_up(n) -> y0 per band
    user_bands = []
    for u in users:
        addr = u["user"]
        ns = amm.read_user_tick_numbers(addr)
        n1, n2 = ns[0], ns[1]
        xs_ys = amm.get_xy(addr)
        xs = list(xs_ys[0])
        ys = list(xs_ys[1])
        bands = []
        # one entry per band, in order n1 .. n2
        for i, n in enumerate(range(n1, n2 + 1)):
            p_o_up = p_oracle_up_local(n)
            x = xs[i]
            y = ys[i]
            y0 = get_y0(x, y, p_o_now, p_o_up, A)
            bands.append({"n": n, "p_o_up": p_o_up, "x": x, "y": y, "y0": y0})
        user_bands.append({**u, "bands": bands, "n1": n1, "n2": n2})
        print(f"  {addr[:10]}.. debt={u['debt']/WAD:.2f}  N={n2-n1+1}  "
              f"x={u['x']/WAD:.2f}  y={u['y']/WAD:.2f}  health={u['health']/WAD:.4f}")

    # Sanity: at p = p_o_now, sum_band(xy_at_price) should ~ match user's stored x, y
    print("\nSanity check at current oracle price:")
    for u in user_bands[:3]:
        sx = sy = 0.0
        for b in u["bands"]:
            x_b, y_b = xy_at_price(b["y0"], b["p_o_up"], p_o_now, A)
            sx += x_b
            sy += y_b
        print(f"  {u['user'][:10]}..  computed x={sx/WAD:.2f} y={sy/WAD:.4f}  "
              f"vs stored x={u['x']/WAD:.2f} y={u['y']/WAD:.4f}")

    # Sweep CRV price.
    p_min = 0.01
    p_max = 1.3
    # Compute past the visible window so curves don't terminate visibly
    # short of the right edge of the frame.
    p_compute_max = 1.5
    n_pts = 460
    prices = np.linspace(p_min, p_compute_max, n_pts)

    fig, ax = plt.subplots(figsize=(11, 7))

    total_debt = sum(u["debt"] for u in user_bands)

    # Per-user value curves so we can also draw the per-position envelope.
    nonzero = [u for u in user_bands if u["debt"] > 0]
    per_user_value = np.zeros((len(nonzero), len(prices)))
    for ui, u in enumerate(nonzero):
        for i, p in enumerate(prices):
            p_wad = p * WAD
            v = 0.0
            for b in u["bands"]:
                xb, yb = xy_at_price(b["y0"], b["p_o_up"], p_wad, A)
                v += xb + p_wad * yb / WAD
            per_user_value[ui, i] = v

    debts = np.array([float(u["debt"]) for u in nonzero])
    # Once a position's value reaches its debt, a profitable hard-liquidation
    # would close it: its contribution to the pool stays capped at debt and
    # it stops accruing further upside. Since value(p) is monotonic in p,
    # this is equivalent to clipping at debt.
    per_user_capped = np.minimum(per_user_value, debts[:, None])
    total_value = per_user_capped.sum(axis=0)
    per_user_solvency = per_user_value / debts[:, None] * 100.0
    # Only paint the envelope from non-dust positions: dust positions can
    # have wildly off-scale solvency curves that aren't useful here.
    DUST_DEBT = 1000 * WAD
    big_mask = debts >= DUST_DEBT
    n_big = int(big_mask.sum())
    if n_big > 0:
        big_debts = debts[big_mask]
        big_solvency = per_user_solvency[big_mask]
        total_big = big_debts.sum()
        # A position is "alive" only while its uncapped solvency is below
        # 100%. Once it hits 100% it gets profitably liquidated and stops
        # contributing to the envelope at any higher price.
        big_solvency_active = np.where(big_solvency < 100.0,
                                       big_solvency, np.nan)
        env_low = np.nanmin(big_solvency_active, axis=0)
        env_high = np.nanmax(big_solvency_active, axis=0)

        # Inside the envelope we paint a base translucent floor and stack
        # one fill per position, going from the bottom of the envelope up
        # to that position's solvency curve. Each fill's alpha is
        # proportional to that position's debt share, so the composited
        # alpha at any (p, y) tracks the debt-weighted fraction of
        # positions lying *above* y at that price — i.e. the share of bad
        # debt that's actually liquidatable from this y level upward.
        # Alpha goes from ~20% at the top of the envelope (almost nothing
        # left to liquidate above) to ~80% at the bottom.
        ax.fill_between(prices, env_low, env_high,
                        color=NEWSPAPER_YELLOW, alpha=0.2, zorder=1, linewidth=0)
        # Sum α_i ≈ 1.5 so 1 - exp(-1.5) ≈ 0.78 ⇒ ~80% maximum darkness.
        ALPHA_SCALE = 1.5
        for i in range(n_big):
            a = ALPHA_SCALE * float(big_debts[i] / total_big)
            ax.fill_between(prices, env_low, big_solvency_active[i],
                            color=NEWSPAPER_YELLOW, alpha=a, zorder=1,
                            linewidth=0)
        # Proxy artist for the legend (Patch isn't a real on-axes artist;
        # add it via the legend handles list further down).
        from matplotlib.patches import Patch
        gradient_proxy = Patch(facecolor=NEWSPAPER_YELLOW, alpha=0.5,
                               label=f"liquidatable-debt density "
                                     f"(n={n_big} pos, debt > 1k crvUSD)")
    else:
        gradient_proxy = None

    # Redeemable, redefined: at each price, the highest individual solvency
    # among positions still in the AMM. A position is "still in" iff its
    # uncapped solvency hasn't yet reached 100% (i.e. it hasn't been
    # profitably liquidated). We restrict to non-dust positions for the
    # same reason as the envelope.
    worst_remaining = np.full_like(prices, np.nan)
    big_solvency = per_user_solvency[big_mask]
    for j in range(len(prices)):
        active = big_solvency[:, j] < 100.0
        if active.any():
            worst_remaining[j] = big_solvency[active, j].max()

    # "Fair" solvency: rest of the market is assumed to repay in full,
    # so solvent debt contributes 1:1 to both numerator and denominator.
    solvent_debt = market_debt - total_debt
    fair_pct = (total_value + solvent_debt) / market_debt * 100.0

    ax.plot(prices, worst_remaining, color="black", linewidth=2.4, zorder=4,
            label="next-to-liquidate (top of still-remaining positions)")
    ax.plot(prices, fair_pct, color="darkgreen", linewidth=2.0,
            linestyle="--", zorder=4,
            label="fair solvency (market average)")

    # With per-position liquidation cap, full recovery happens when the
    # *slowest* position reaches 100% — i.e. max over positions of each
    # one's individual break-even price. Restrict to non-dust positions
    # (same filter as the envelope) so handful-of-cents loans don't
    # hijack the break-even price.
    per_user_break_even = []
    for vi, debt_i, big in zip(per_user_value, debts, big_mask):
        if not big:
            continue
        if vi[-1] < debt_i:
            per_user_break_even.append(np.inf)
        elif vi[0] >= debt_i:
            per_user_break_even.append(prices[0])
        else:
            per_user_break_even.append(float(np.interp(debt_i, vi, prices)))
    p_first = min(per_user_break_even) if per_user_break_even else np.inf
    p_full = max(per_user_break_even) if per_user_break_even else np.inf

    def mark(p, text, xytext_offset, color="red"):
        if not (np.isfinite(p) and prices[0] <= p <= prices[-1]):
            return
        ax.plot([p], [100.0], "o", color=color, markersize=7, zorder=20)
        ann = ax.annotate(text, xy=(p, 100.0),
                          xytext=xytext_offset, textcoords="offset points",
                          fontsize=10, color=color, zorder=20,
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", edgecolor=color, lw=0.8),
                          arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        ann.get_bbox_patch().set_zorder(20)

    mark(p_first, f"full recovery starts at p = ${p_first:.3f}",
         (-220, 60))
    mark(p_full, f"full recovery finishes at p = ${p_full:.3f}",
         (-150, 28))

    ax.axhline(100, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="full recovery (100%)")
    ax.axvline(p_o_now / WAD, color="blue", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"current p_o = {p_o_now/WAD:.3f}")
    ax.set_xlabel("CRV price (crvUSD)")
    ax.set_ylabel("Solvency: Σ value / Σ debt   [%]")
    ax.set_title(f"Aggregate solvency vs CRV price across "
                 f"{len(user_bands)} unprofitable positions\n"
                 f"Controller: {CONTROLLER}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(p_min, p_max)
    handles, labels = ax.get_legend_handles_labels()
    if gradient_proxy is not None:
        handles = [gradient_proxy] + handles
        labels = [gradient_proxy.get_label()] + labels
    ax.legend(handles, labels, loc="lower right", fontsize=10)
    ax.set_ylim(top=115)

    out = "/home/michwill/Projects/stableswap-tools/plots/recovery_vs_crv_price.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nSaved plot: {out}")
    plt.show()


if __name__ == "__main__":
    main()
