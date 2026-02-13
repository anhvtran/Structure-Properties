#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata

def read_xyz(filename):
    """Read XYZ file and extract atomic positions and lattice vectors."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())
    line = lines[1]
    start = line.find('Lattice="') + 9
    end = line.find('"', start)
    lattice_str = line[start:end]
    lattice_vals = np.array([float(x) for x in lattice_str.split()])
    t = lattice_vals.reshape(3, 3)
    
    atoms = []
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atom_type = parts[0]
        if atom_type == 'C':
            atype = 1
        elif atom_type == 'Mo':
            atype = 2
        elif atom_type == 'S':
            atype = 3
        else:
            atype = 0
        atoms.append([x, y, z, atype])
    return np.array(atoms), t

def setup_matplotlib_style(fig_width=16/2.54, aspect_ratio=0.8):
    """Configure matplotlib style for consistent plots."""
    fig_height = fig_width * aspect_ratio
    
    mpl.rcParams.update({
        "axes.linewidth": 0.8,
        "axes.labelsize": 12,
        "font.size": 12,
        "font.weight": "normal",
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "mathtext.default": "regular",
        "figure.figsize": (fig_width, fig_height),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    return fig_width, fig_height

def plot_interlayer_distance(filename, 
                             interp_method='cubic',
                             grid_points=100,
                             nm_extend=1,
                             nm_plot=1,
                             output=None,
                             show_plot=True):
    
    fig_width, fig_height = setup_matplotlib_style()
    
    # ========== READ FILE ==========
    print(f'Reading file: {filename}')
    a, t = read_xyz(filename)
    
    t1 = np.array([t[0, 0], t[0, 1], 0])
    t2 = np.array([t[1, 0], t[1, 1], 0])
    a = a[:, :3]
    print(f't1: {t1[:2]}')
    print(f't2: {t2[:2]}')

    # Center atoms around z=0
    a[:, 2] = a[:, 2] - (np.max(a[:, 2]) + np.min(a[:, 2])) / 2
    id_top = a[:, 2] > 0
    at = a[id_top]
    id_bot = a[:, 2] < 0
    ab = a[id_bot]
    
    # Extend atoms for periodic boundary conditions
    ab_ext = []
    at_ext = []
    
    for j in range(-nm_extend, nm_extend + 1):
        for i in range(-nm_extend, nm_extend + 1):
            ab_ext.append(ab + i * t1 + j * t2)
            at_ext.append(at + i * t1 + j * t2)
    
    ab_ext = np.vstack(ab_ext)
    at_ext = np.vstack(at_ext)
    
    # Create interpolation grid
    frac = np.linspace(0, 1, grid_points)
    x_grid = []
    y_grid = []
    
    for i in range(grid_points):
        for j in range(grid_points):
            pos = frac[i] * t1 + frac[j] * t2
            x_grid.append(pos[0])
            y_grid.append(pos[1])
    
    x_grid = np.array(x_grid)
    y_grid = np.array(y_grid)
    
    # ========== INTERPOLATION ==========
    print(f'Interpolating with method: {interp_method}')
    z_bot = griddata(ab_ext[:, :2], ab_ext[:, 2], 
                     (x_grid, y_grid), method=interp_method)
    z_top = griddata(at_ext[:, :2], at_ext[:, 2], 
                     (x_grid, y_grid), method=interp_method)
    d_local = z_top - z_bot
    
    # ========== STATISTICS ==========
    d_valid = d_local[~np.isnan(d_local)]
    print(f'\nInterlayer Distance Statistics:')
    print(f'  Min:      {np.min(d_valid):.4f} Å')
    print(f'  Max:      {np.max(d_valid):.4f} Å')
    print(f'  Mean:     {np.mean(d_valid):.4f} Å')
    print(f'  Std Dev:  {np.std(d_valid):.4f} Å')
    print(f'  Range:    {np.max(d_valid) - np.min(d_valid):.4f} Å')
    print('=========================================\n')
    
    # Replicate for plotting
    x_plot = []
    y_plot = []
    d_plot = []
    
    for j in range(-nm_plot, nm_plot + 1):
        for i in range(-nm_plot, nm_plot + 1):
            x_plot.append(x_grid + i * t1[0] + j * t2[0])
            y_plot.append(y_grid + i * t1[1] + j * t2[1])
            d_plot.append(d_local)
    
    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)
    d_plot = np.hstack(d_plot)

    # ========== PLOT ==========
    fig, ax = plt.subplots()
    valid_mask = ~np.isnan(d_plot)
    x_valid = x_plot[valid_mask]
    y_valid = y_plot[valid_mask]
    d_valid = d_plot[valid_mask]

    sort_idx = np.argsort(d_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]
    d_sorted = d_valid[sort_idx]

    scatter = ax.scatter(x_sorted, y_sorted, c=d_sorted, s=7, 
                        cmap='viridis', edgecolors='none')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('d (Å)', fontsize=14)
    
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=600, bbox_inches='tight')
        print(f'Saved to: {output}')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_buckling(filename, 
                  mode=2,
                  nm_extend=1,
                  output=None,
                  show_plot=True):
    
    fig_width, fig_height = setup_matplotlib_style()
    
    # ========== READ FILE ==========
    print(f'Reading file: {filename}')
    a, t = read_xyz(filename)
    
    t1 = np.array([t[0, 0], t[0, 1], 0])
    t2 = np.array([t[1, 0], t[1, 1], 0])
    a = a[:, :3]
    print(f't1: {t1[:2]}')
    print(f't2: {t2[:2]}')
    
    # ========== PROCESS BASED ON MODE ==========
    if mode == 1:
        # Fixed bottom layer
        a[:, 2] = a[:, 2] - a[0, 2] - 1
        id_top = a[:, 2] > 0
        at = a[id_top]
        id_bot = a[:, 2] < 0
        ab = a[id_bot]
        a = at
        a[:, 2] = a[:, 2] + 1
        print('Mode: Fixed bottom layer')
        
    elif mode == 2:
        # Fully free relaxation
        a[:, 2] = a[:, 2] - (np.max(a[:, 2]) + np.min(a[:, 2])) / 2
        id_top = a[:, 2] > 0
        at = a[id_top]
        id_bot = a[:, 2] < 0
        ab = a[id_bot]
        a = at
        a[:, 2] = a[:, 2] * 2
        print('Mode: Fully free relaxation')
        # Uncomment below for relative buckling:
        # at[:, 2] = at[:, 2] - np.mean(at[:, 2])
        # ab[:, 2] = ab[:, 2] - np.mean(ab[:, 2])
    
    r1 = a.copy()
    
    # Extend for periodic boundaries
    r1_ext = []
    for j in range(-nm_extend, nm_extend + 1):
        for i in range(-nm_extend, nm_extend + 1):
            r1_ext.append(r1 + i * t1 + j * t2)
    
    r1_ext = np.vstack(r1_ext)
    
    # Sort by z coordinate
    r1_ext = r1_ext[np.argsort(r1_ext[:, 2])]
    
    # ========== STATISTICS ==========
    print(f'\nBuckling Statistics:')
    print(f'  Min z:    {np.min(r1_ext[:, 2]):.4f} Å')
    print(f'  Max z:    {np.max(r1_ext[:, 2]):.4f} Å')
    print(f'  Mean z:   {np.mean(r1_ext[:, 2]):.4f} Å')
    print(f'  Range:    {np.max(r1_ext[:, 2]) - np.min(r1_ext[:, 2]):.4f} Å')
    print('=========================================\n')
    
    # ========== PLOT ==========
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    scatter = ax.scatter(r1_ext[:, 0], r1_ext[:, 1], 
                        c=r1_ext[:, 2], s=30, 
                        cmap='viridis', edgecolors='none')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('z (Å)', fontsize=14)
    
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=600, bbox_inches='tight')
        print(f'Saved to: {output}')
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    
    # Configuration
    fname0 = 'ft_me_n1m30'
    filename = f'./test/opt_{fname0}.xyz'
    
    # Choose which plot to generate
    plot_type = 'raw'  # 'interlayer' or 'buckling' or 'both'
    
    if plot_type in ['interp', 'both']:
        plot_interlayer_distance(
            filename=filename,
            interp_method='cubic',
            grid_points=100,
            nm_extend=1,
            nm_plot=1,
            output=f'interlayer_dist_{fname0}.png',
            show_plot=True
        )
    
    if plot_type in ['raw', 'both']:
        plot_buckling(
            filename=filename,
            mode=2,  # 1: fixed bottom, 2: fully free
            nm_extend=1,
            output=f'buckling_{fname0}.png',
            show_plot=True
        )
