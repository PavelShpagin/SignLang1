# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import numpy as np
import os

import sys
BOX_COLOR = 'black'
plt.rcParams['font.family'] = 'Segoe UI Emoji'
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
TITLE_STYLE = dict(fontsize=16, fontweight='bold', color=BOX_COLOR)

def add_gesture(ax, emoji, center, fontsize, w_label, h_label, title):
    t = ax.text(center[0], center[1], emoji, fontsize=fontsize, ha='center', va='center', clip_on=False)
    ax.set_aspect('equal')
    ax.axis('off')
    return t, (w_label, h_label, title)

# FIST (w/h = 2.0, ratio range 1.5-2.5)
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
t1, (w1, h1, title1) = add_gesture(ax1, '\U0001F44A', (5, 5), 100, '8', '4', 'FIST')

# PALM (w/h = 1.0, ratio range 0.9-1.1)
ax2 = axes[1]
ax2.set_xlim(0, 22)
ax2.set_ylim(0, 26)
t2, (w2, h2, title2) = add_gesture(ax2, '\U0001F590', (11, 13), 140, '16', '16', 'PALM')

def get_glyph_image(char, fontsize=200):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.text(0.5, 0.5, char, fontsize=fontsize, ha='center', va='center', fontfamily='Segoe UI Emoji')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    img = imread(buf)
    
    # Crop to content
    # Assuming white background (1,1,1) or transparent
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
    else:
        gray = img.mean(axis=2)
        rows = np.any(gray < 0.9, axis=1)
        cols = np.any(gray < 0.9, axis=0)
        
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return img[ymin:ymax+1, xmin:xmax+1]
    return img

# VICTORY SIGN - use image-based approach for perfect centering
ax3 = axes[2]
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 28)
ax3.axis('off')

glyph_img = get_glyph_image('\u270C') # Use standard Victory Hand
# Center at (10, 14). Box is 10x20.
h_img, w_img = glyph_img.shape[:2]
aspect = h_img / w_img
    
# Victory sign box (match labels and report)
box_w = 10.0
box_h = 16.0

# Center: x=10, y=14
cx, cy = 10.0, 14.0
x0 = cx - box_w / 1.3
x1 = cx + box_w / 1.3
y0 = cy + box_h / 2
y1 = cy - box_h / 2

ax3.imshow(glyph_img, extent=[x0, x1, y1, y0], aspect='equal', interpolation='bilinear')
t3, w3, h3, title3 = None, '10', '12', 'VICTORY SIGN'

# Render to get text extents, then draw tight boxes
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def draw_tight_box(ax, text_obj, w_label, h_label, title):
    bbox = text_obj.get_tightbbox(renderer=renderer) or text_obj.get_window_extent(renderer=renderer)
    bbox_data = bbox.transformed(ax.transData.inverted())
    x0, y0 = bbox_data.x0, bbox_data.y0
    w = bbox_data.width
    h = bbox_data.height

    w_inset = w * 0.12
    h_inset = h * 0.05
    x0 += w_inset
    y0 += h_inset
    w -= 2 * w_inset
    h -= 2 * h_inset

    rect = patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=BOX_COLOR, linewidth=4, linestyle='--')
    ax.add_patch(rect)

    pad = 0.8
    y_arrow = y0 - pad
    ax.annotate('', xy=(x0, y_arrow), xytext=(x0 + w, y_arrow),
                arrowprops=dict(arrowstyle='<->', color=BOX_COLOR, lw=2))
    ax.text(x0 + w/2, y_arrow - 0.6, f'W = {w_label} cm', ha='center', va='top', fontsize=14, color=BOX_COLOR, fontweight='bold')

    x_arrow = x0 + w + pad
    ax.annotate('', xy=(x_arrow, y0), xytext=(x_arrow, y0 + h),
                arrowprops=dict(arrowstyle='<->', color=BOX_COLOR, lw=2))
    ax.text(x_arrow + 0.4, y0 + h/2, f'H = {h_label} cm', ha='left', va='center', fontsize=14, color=BOX_COLOR, fontweight='bold')


def draw_fixed_box(ax, x_center, y_center, w, h, w_label, h_label, title):
    # Calculate corner from center and dimensions
    x0 = x_center - w/2
    y0 = y_center - h/2
    
    rect = patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=BOX_COLOR, linewidth=4, linestyle='--')
    ax.add_patch(rect)

    pad = 0.8
    y_arrow = y0 - pad
    ax.annotate('', xy=(x0, y_arrow), xytext=(x0 + w, y_arrow), arrowprops=dict(arrowstyle='<->', color=BOX_COLOR, lw=2))
    ax.text(x0 + w/2, y_arrow - 0.6, f'W = {w_label} cm', ha='center', va='top', fontsize=14, color=BOX_COLOR, fontweight='bold')

    x_arrow = x0 + w + pad
    ax.annotate('', xy=(x_arrow, y0), xytext=(x_arrow, y0 + h), arrowprops=dict(arrowstyle='<->', color=BOX_COLOR, lw=2))
    ax.text(x_arrow + 0.4, y0 + h/2, f'H = {h_label} cm', ha='left', va='center', fontsize=14, color=BOX_COLOR, fontweight='bold')
    # Title is drawn separately for consistent alignment across subplots

draw_tight_box(ax1, t1, w1, h1, title1)
draw_tight_box(ax2, t2, w2, h2, title2)
draw_fixed_box(ax3, 10, 14, box_w, box_h, w3, h3, title3)

fig.subplots_adjust(left=0.08, right=0.96, bottom=0.12, top=0.9, wspace=0.25)

# Draw titles at identical vertical level (figure coords)
y_title = 0.92
for ax, title in zip(axes, [title1, title2, title3]):
    pos = ax.get_position()
    x_center = pos.x0 + pos.width / 2
    fig.text(x_center, y_title, title, ha='center', va='bottom', transform=fig.transFigure, **TITLE_STYLE)

plt.savefig('box_examples.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
print("Розміри: КУЛАК 8x4cm (w/h=2.0), ДОЛОНЯ 16x16cm (w/h=1.0), ЗНАК ПЕРЕМОГИ 10x12cm (h/w=1.2)")
