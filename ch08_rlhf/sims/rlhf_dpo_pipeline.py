"""RLHF vs DPO pipeline comparison diagram for Chapter 8.
Uses DAG-style drawing helpers (cf. ch09 identification_dags.py)."""

import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS
apply_style()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Drawing utilities (DAG-style helpers for rectangular process nodes)
# ---------------------------------------------------------------------------

RECT_PAD = 0.15        # rounding pad for FancyBboxPatch
ARROW_LW = 1.4
DASH_STYLE = (0, (5, 4))


def _rect_edge_point(center, half_w, half_h, target, pad=RECT_PAD):
    """Return the point on the boundary of a rounded rectangle (at *center*
    with half-extents *half_w*, *half_h*) closest to *target*, along the
    line from center to target.  Used to clip arrows at box edges."""
    cx, cy = center
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return center
    # Scale factors to hit each edge
    sx = (half_w + pad) / abs(dx) if abs(dx) > 1e-9 else 1e9
    sy = (half_h + pad) / abs(dy) if abs(dy) > 1e-9 else 1e9
    s = min(sx, sy)
    return (cx + s * dx, cy + s * dy)


def draw_rect_node(ax, xy, label, half_w=0.9, half_h=0.35,
                   facecolor='white', edgecolor='black', linewidth=1.4,
                   linestyle='-', fontsize=11, textcolor='black',
                   alpha=1.0, zorder=3):
    """Draw a rounded-rectangle process node centred at *xy*."""
    box = mpatches.FancyBboxPatch(
        (xy[0] - half_w, xy[1] - half_h), 2 * half_w, 2 * half_h,
        boxstyle=f'round,pad={RECT_PAD}',
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, linestyle=linestyle,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], label, ha='center', va='center',
            fontsize=fontsize, color=textcolor, zorder=zorder + 1)


def draw_edge(ax, p1, p2, hw1=0.9, hh1=0.35, hw2=0.9, hh2=0.35,
              dashed=False, lw=ARROW_LW, color='black', curve=0.0):
    """Draw a directed arrow between two rectangular nodes, clipping
    start/end to rounded-rect boundaries."""
    start = _rect_edge_point(p1, hw1, hh1, p2)
    end   = _rect_edge_point(p2, hw2, hh2, p1)
    props = dict(arrowstyle='->', lw=lw, color=color, shrinkA=0, shrinkB=0)
    if dashed:
        props['linestyle'] = DASH_STYLE
    if abs(curve) > 1e-6:
        props['connectionstyle'] = f'arc3,rad={curve}'
    ax.annotate('', xy=end, xytext=start, arrowprops=props, zorder=2)


def draw_text_arrow(ax, origin, target, hw_tgt=0.9, hh_tgt=0.35,
                    label=None, label_offset=(0, 0), lw=ARROW_LW,
                    color='black', fontsize=9.5):
    """Arrow from a bare text position (no bounding box) into a rect node."""
    end = _rect_edge_point(target, hw_tgt, hh_tgt, origin)
    props = dict(arrowstyle='->', lw=lw, color=color, shrinkA=0, shrinkB=0)
    ax.annotate('', xy=end, xytext=origin, arrowprops=props, zorder=2)
    if label:
        mx = (origin[0] + end[0]) / 2 + label_offset[0]
        my = (origin[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=fontsize, color=color, zorder=3)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def generate_outputs():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(False)

    # Layout constants -------------------------------------------------
    Y_TOP = 4.5            # RLHF row centre
    Y_BOT = 1.5            # DPO row centre
    Y_MID = (Y_TOP + Y_BOT) / 2

    X_SFT   = 2.0          # SFT box centre-x
    X_RM    = 6.0           # Reward Model centre-x
    X_FINAL = 10.5          # PPO / DPO optimisation centre-x

    # Shared box dimensions
    HW_STD = 1.05           # half-width standard boxes
    HH_STD = 0.38           # half-height
    HW_WIDE = 1.35          # half-width for PPO / DPO boxes
    HW_RM = 1.15            # half-width for reward model

    # Light background tints (15 % alpha on palette colours)
    FILL_SFT   = '#e0e0e0'
    FILL_RM    = '#f0dcc0'     # warm: reward model
    FILL_PPO   = '#dcc0c0'     # red tint: PPO
    FILL_DPO   = '#c0d4e8'     # blue tint: DPO
    FILL_GHOST = '#f0f0f0'

    EDGE_CLR = COLORS['black']

    # ==================================================================
    # Row labels
    # ==================================================================
    ax.text(-0.1, Y_TOP, 'RLHF', ha='center', va='center',
            fontsize=13, fontweight='bold', color=COLORS['black'])
    ax.text(-0.1, Y_BOT, 'DPO', ha='center', va='center',
            fontsize=13, fontweight='bold', color=COLORS['black'])

    # ==================================================================
    # TOP ROW: RLHF pipeline
    # ==================================================================
    sft_top = (X_SFT, Y_TOP)
    rm_top  = (X_RM, Y_TOP)
    ppo_top = (X_FINAL, Y_TOP)

    draw_rect_node(ax, sft_top, r'SFT Model $\pi_{\mathrm{ref}}$',
                   half_w=HW_STD, half_h=HH_STD,
                   facecolor=FILL_SFT, edgecolor=EDGE_CLR)

    draw_rect_node(ax, rm_top, r'Reward Model $r_\varphi$',
                   half_w=HW_RM, half_h=HH_STD,
                   facecolor=FILL_RM, edgecolor=EDGE_CLR)

    draw_rect_node(ax, ppo_top, r'PPO Fine-tuning',
                   half_w=HW_WIDE, half_h=HH_STD,
                   facecolor=FILL_PPO, edgecolor=EDGE_CLR)

    # Arrows: SFT -> RM -> PPO
    draw_edge(ax, sft_top, rm_top,
              hw1=HW_STD, hh1=HH_STD, hw2=HW_RM, hh2=HH_STD,
              color=EDGE_CLR)
    draw_edge(ax, rm_top, ppo_top,
              hw1=HW_RM, hh1=HH_STD, hw2=HW_WIDE, hh2=HH_STD,
              color=EDGE_CLR)

    # Human Preferences label above RM
    pref_top = (X_RM, Y_TOP + 1.15)
    ax.text(pref_top[0], pref_top[1], 'Human\nPreferences',
            ha='center', va='center', fontsize=9.5, color=COLORS['black'])
    draw_text_arrow(ax, pref_top, rm_top,
                    hw_tgt=HW_RM, hh_tgt=HH_STD, color=EDGE_CLR)

    # PPO inner-loop annotation below the PPO box
    ax.text(X_FINAL, Y_TOP - 1.1,
            r'$\pi_\theta$ generates $\;\to\; r_\varphi$ scores'
            '\n'
            r'$\to\; \lambda_{\mathrm{KL}}$ penalty $\;\to\;$ update',
            ha='center', va='center', fontsize=7.5, color=COLORS['gray'])

    # Curved feedback arrow looping over the PPO box
    loop_top = Y_TOP + HH_STD + RECT_PAD + 0.04
    fb_start = (X_FINAL + HW_WIDE * 0.6, loop_top)
    fb_end   = (X_FINAL - HW_WIDE * 0.6, loop_top)
    ax.annotate('', xy=fb_end, xytext=fb_start,
                arrowprops=dict(arrowstyle='->', lw=1.2,
                                color=COLORS['red'],
                                connectionstyle='arc3,rad=0.45',
                                shrinkA=0, shrinkB=0),
                zorder=2)

    # ==================================================================
    # BOTTOM ROW: DPO pipeline
    # ==================================================================
    sft_bot   = (X_SFT, Y_BOT)
    ghost_bot = (X_RM, Y_BOT)
    dpo_bot   = (X_FINAL, Y_BOT)

    draw_rect_node(ax, sft_bot, r'SFT Model $\pi_{\mathrm{ref}}$',
                   half_w=HW_STD, half_h=HH_STD,
                   facecolor=FILL_SFT, edgecolor=EDGE_CLR)

    # Ghost reward model: dashed border, light gray fill, gray text
    draw_rect_node(ax, ghost_bot, r'Reward Model',
                   half_w=HW_RM, half_h=HH_STD,
                   facecolor=FILL_GHOST, edgecolor=COLORS['gray'],
                   linewidth=1.2, linestyle='--',
                   textcolor=COLORS['gray'], alpha=0.55, fontsize=10)

    draw_rect_node(ax, dpo_bot,
                   r'Direct Optimization $\mathcal{L}_{\mathrm{DPO}}$',
                   half_w=HW_WIDE, half_h=HH_STD,
                   facecolor=FILL_DPO, edgecolor=EDGE_CLR, fontsize=10.5)

    # Curved bypass arrow: SFT -> DPO, arcing below the ghost box
    bypass_start = _rect_edge_point(sft_bot, HW_STD, HH_STD, dpo_bot)
    bypass_end   = _rect_edge_point(dpo_bot, HW_WIDE, HH_STD, sft_bot)
    ax.annotate('', xy=bypass_end, xytext=bypass_start,
                arrowprops=dict(arrowstyle='->', lw=1.5, color=EDGE_CLR,
                                connectionstyle='arc3,rad=-0.22',
                                shrinkA=0, shrinkB=0),
                zorder=2)

    # Human Preferences label above DPO box
    pref_bot = (X_FINAL, Y_BOT + 1.15)
    ax.text(pref_bot[0], pref_bot[1], 'Human\nPreferences',
            ha='center', va='center', fontsize=9.5, color=COLORS['black'])
    draw_text_arrow(ax, pref_bot, dpo_bot,
                    hw_tgt=HW_WIDE, hh_tgt=HH_STD, color=EDGE_CLR)

    # ==================================================================
    # Between rows: implicit reward formula
    # ==================================================================
    ax.text((X_SFT + X_FINAL) / 2, 2.7,
            r'Implicit reward:  $r(x,y) = \beta \log\left(\pi_\theta(y\mid x)'
            r'\,/\,\pi_{\mathrm{ref}}(y\mid x)\right)$',
            ha='center', va='center', fontsize=11, color=COLORS['black'],
            style='italic')

    # Vertical dashed line connecting the two SFT boxes
    ax.plot([X_SFT, X_SFT],
            [Y_TOP - HH_STD - RECT_PAD - 0.08,
             Y_BOT + HH_STD + RECT_PAD + 0.08],
            linestyle='--', color=COLORS['gray'], lw=1.0, alpha=0.5, zorder=1)

    # ==================================================================
    # Save
    # ==================================================================
    out_path = os.path.join(os.path.dirname(__file__), 'rlhf_dpo_pipeline.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true',
                        help='No computation to cache (diagram-only script)')
    parser.add_argument('--plots-only', action='store_true',
                        help='Runs normally (same as no flags)')
    args = parser.parse_args()
    if args.data_only:
        print("No computation to cache (diagram-only script).")
        sys.exit(0)
    generate_outputs()
