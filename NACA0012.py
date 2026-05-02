#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:11:59 2025

@author: himalayagaur
"""

""" PiNNs (Physics informed Neural Network model using pytorch) for steady incompressible 2D flow around a NACA0012 airfoil
- Uses a fully connected MLP to predict u, v, p from (x, y, AoA, Re)
- Enforces continuity and momentum residuals at collocation points using autograd
- Enforces boundary conditions: no-slip on airfoil surface, far-field freestream
- Computes surface pressure coefficient Cp and lift coefficient Cl (pressure only)
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.path import Path


# --------Geometry----------

def naca4_coordinates(naca='0012', n_pts=200, chord=1.0):
    """ Simple NACA 4 series generator (supports symmetric and cambered) """
    if len(naca) != 4:
        raise ValueError('Expect 4-digit NACA code (e.g 0012)')
    
    # parse digits
    m = int(naca[0]) / 100.0
    p = int(naca[1]) / 10.0
    t = int(naca[2:]) / 100.0

    # cosine spacing
    beta = np.linspace(0.0, math.pi, n_pts)
    x = 0.5 * (1.0 - np.cos(beta)) * chord

    # thickness distribution
    yt = 5 * t * (
        0.2969 * np.sqrt(x/chord)
        - 0.1260 * (x/chord)
        - 0.3516 * (x/chord)**2
        + 0.2843 * (x/chord)**3
        - 0.1036 * (x/chord)**4
    )

    # camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m > 0:
        for i, xi in enumerate(x):
            xr = xi / chord
            if xr < p:
                yc[i] = m/(p**2) * (2*p*xr - xr**2)
                dyc_dx[i] = 2*m/(p**2) * (p - xr)
            else:
                yc[i] = m/((1-p)**2) * ((1-2*p) + 2*p*xr - xr**2)
                dyc_dx[i] = 2*m/((1-p)**2) * (p - xr)

    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # assemble polygon around airfoil
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    return x_coords, y_coords
        

# --------- Domain -------------
def make_domain(bounds, n_samples = 50000, foil_path=None):
    """ Sample points in rectangular domain bounds = [xmin, xmax, ymin, ymax] and remove
    points inside the airfoil polygon using matplotlib.path. path returns Nx2 array points
    """
    
    xmin, xmax, ymin, ymax = bounds
    # sample uniformly
    xs = np.random.uniform(xmin, xmax, size=n_samples)
    ys = np.random.uniform(ymin, ymax, size=n_samples)
    pts = np.vstack([xs, ys]).T
    if foil_path is not None:
        path = Path(foil_path)
        mask = ~path.contains_points(pts)
        pts = pts[mask]
    return pts

# --------Sample Boundary On Foil -------------

def sample_boundary_on_foil(foil_xy, n_bd=500):
    """Uniformly sample points along foil surface polygon (Closed)
    Returns:
        pts: (N, 2) array of sampled points
        normals: (N, 2) array of outward normals (unit vectors)
    """
    x = foil_xy[:, 0]
    y = foil_xy[:, 1]
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    seg_len = np.sqrt(dx ** 2 + dy ** 2)
    cum = np.cumsum(seg_len)
    total = cum[-1]

    # sample positions along the perimeter
    t = np.linspace(0, total, n_bd, endpoint=False)

    pts = []
    normals = []

    i = 0
    for val in t:
        while val > cum[i]:
            i += 1

        seg_start = cum[i-1] if i > 0 else 0
        frac = (val - seg_start) / seg_len[i]

        x0, y0 = x[i], y[i]
        x1, y1 = x[(i + 1) % len(x)], y[(i + 1) % len(x)]
        px = x0 + frac * (x1 - x0)
        py = y0 + frac * (y1 - y0)
        pts.append((px, py))

        # tangent
        tx = x1 - x0
        ty = y1 - y0
        # outward normal (rotate tangent 90 deg CCW)
        nx = ty
        ny = -tx
        nrm = math.hypot(nx, ny) + 1e-12
        normals.append((nx / nrm, ny / nrm))

    return np.array(pts), np.array(normals)      

# -------------- PiNN Model ---------------

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
            ])
        # Xavier init
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, x):
        # x: [N, input_dim] (x, y, AoA, Re_norm optional)
        a = x
        for i in range(len(self.layers)-1):
            a = torch.tanh(self.layers[i](a))
        out = self.layers[-1](a)
        return out
        
# ------------ Physics Helper -------------

def gradients(u, x, order=1):
    """ Compute derivatives of u w.r.t x using autograd. x can be tensor with shape [N,d]
    if order == 1 return du/dx (same shape as u) or if order==2 returns second derivative
    This helper assumes u is [N, 1] and x is [N, d]
    """
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    if order == 1:
        return grads
    elif order == 2:
        # Compute Jacobian of grads w.r.t. x again
        outs = []
        for i in range(grads.shape[1]):
            g_i = grads[:, i:i+1]
            g2 = torch.autograd.grad(g_i, x, grad_outputs=torch.ones_like(g_i), create_graph=True)[0]
            outs.append(g2)
        # outs is list of [N,d] each -- stack --> [d, N, d]
        return torch.stack(outs, dim=1)
    else:
        raise NotImplementedError
        
        
# ---------------- Loss Function ------------------

def loss_pinn(model, collocation_xy, bd_foil_xy, bd_far_xy, U_inf, nu, device):
    # Collocation points: torch tensor [No. 2]
    collocation_xy = collocation_xy.to(device).requires_grad_(True)
    x = collocation_xy
    uvp = model(x)
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]


    # First derivative
    grads_u = gradients(u, x, order=1)  #[N,2] ---> [du/dx, du/dy]
    grads_v = gradients(v, x, order=1)
    grads_p = gradients(p, x, order=1)
    
    
    u_x = grads_u[:, 0:1]
    u_y = grads_u[:, 1:2]
    v_x = grads_v[:, 0:1]
    v_y = grads_v[:, 1:2]
    p_x = grads_p[:, 0:1]
    p_y = grads_p[:, 1:2]
    
    
    
    # Second Derivative (Laplacian)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 0:1]
    
    
    # Continuity Residual
    
    cont = u_x + v_y
    
    # Momentum residuals (steady, incompressible, nondimensionalized if desired)
    mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    
    
    # PDE Loss
    L_pde = (cont ** 2).mean() + (mom_u ** 2).mean() + (mom_v ** 2).mean()
    
    
    # Boundary Losses
    # Foil surface (no-slip) : u = v = 0 on surface
    bd_foil_xy = bd_foil_xy.to(device)
    uvp_bf = model(bd_foil_xy)
    u_bf = uvp_bf[:, 0:1]
    v_bf = uvp_bf[:, 1:2]
    L_b_foil = ((u_bf) ** 2).mean() + ((v_bf) ** 2).mean()
    
    
    # far-field: enforce freestream (rotate freestream according to AoA if AoA included)
    bd_far_xy = bd_far_xy.to(device)
    uvp_ff = model(bd_far_xy)
    u_ff = uvp_ff[:, 0:1]
    v_ff = uvp_ff[:, 1:2]
    # freestream aligned with +x: U_inf, 0
    L_b_far = ((u_ff - U_inf) ** 2).mean() + (v_ff ** 2).mean()
    
    
    # Optionally enforce a pressure reference (To remove constant pressure nullspace)
    p_ref = model(torch.tensor([[10.0, 0.0]], dtype=torch.float32, device=device))[:, 2]
    L_pref = (p_ref ** 2).mean()
    
    # total boundary loss
    L_bc = L_b_foil + L_b_far + 1e-3 * L_pref
    
    return L_pde, L_bc
        



# -------------------------- training driver ----------------------------------



def train_pinn(
    n_epochs=2000,
    device='cpu',
    n_collocation=20000,
    n_bd_foil=800,
    n_bd_far=800,
    show_every=200
):
    device = torch.device(device)

    # Geometry: NACA0012 chord =1 centered at origin
    x_f, y_f = naca4_coordinates('0012', n_pts=400, chord=1.0)
    foil_xy = np.vstack([x_f - 0.25, y_f]).T  # shift so quarter-chord at x = 0

    # Domain bounds (simple rectangle around foil)
    bounds = [-3.0, 8.0, -4.0, 4.0]

    # Sample collocation points (reject inside foil)
    collocation = make_domain(bounds, n_samples=n_collocation * 3, foil_path=foil_xy)
    if collocation.shape[0] > n_collocation:
        idx = np.random.choice(collocation.shape[0], n_collocation, replace=False)
        collocation = collocation[idx]

    # Sample boundary points on foil and far-field
    bd_foil_pts, _ = sample_boundary_on_foil(foil_xy, n_bd_foil)

    # far-field --> sample rectangle boundary points
    xs = np.linspace(bounds[0], bounds[1], n_bd_far)
    top = np.vstack([xs, np.full_like(xs, bounds[3])]).T
    bottom = np.vstack([xs, np.full_like(xs, bounds[2])]).T
    ys = np.linspace(bounds[2], bounds[3], n_bd_far)
    left = np.vstack([np.full_like(ys, bounds[0]), ys]).T
    right = np.vstack([np.full_like(ys, bounds[1]), ys]).T
    bd_far = np.vstack([top, bottom, left, right])

    # convert to torch
    collocation_t = torch.tensor(collocation, dtype=torch.float32, device=device)
    bd_foil_t = torch.tensor(bd_foil_pts, dtype=torch.float32, device=device)
    bd_far_t = torch.tensor(bd_far, dtype=torch.float32, device=device)

    # NonDimensionalization / params
    U_inf = 1.0
    Re = 1e5  # Choose manageable Re (PiNNs struggles at very high Re)
    nu = U_inf / Re

    # Create Model: input (x,y)  ---> output (u,v,p)
    model = PINN([2, 128, 128, 128, 128, 3]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {'loss_pde': [], 'loss_bc': [], 'total': []}

    # Training loop (Adam + LBFGS)
    for ep in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        L_pde, L_bc = loss_pinn(model, collocation_t, bd_foil_t, bd_far_t, U_inf, nu, device)
        loss = L_pde + 10.0 * L_bc
        loss.backward()
        optimizer.step()

        history['loss_pde'].append(L_pde.item())
        history['loss_bc'].append(L_bc.item())
        history['total'].append(loss.item())

        if ep % show_every == 0 or ep == 1:
            print(
                f"Epoch {ep}/{n_epochs} Loss_pde={L_pde.item():.3e} "
                f"Loss_bc={L_bc.item():.3e} Total={loss.item():.3e}"
            )

    # Fine tuning with LBFGS
    print('Finished Adam stage. Run LBFGs...')

    # Evaluate on grid and compute Cp on the airfoil
    model.eval()
    # grid for visualization
    nx, ny = 240, 160
    xs = np.linspace(bounds[0], bounds[1], nx)
    ys = np.linspace(bounds[2], bounds[3], ny)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.vstack([X.ravel(), Y.ravel()]).T
    grid_t = torch.tensor(grid_pts.astype(np.float32), device=device)

    with torch.no_grad():
        uvp = model(grid_t).cpu().numpy()
    U = uvp[:, 0].reshape(Y.shape)
    V = uvp[:, 1].reshape(Y.shape)
    P = uvp[:, 2].reshape(Y.shape)

    # Compute Cp on foil: Cp = (p - p_inf)/(0.5 rho U_inf^2)
    # here rho = 1, U_inf=1 --> Cp = p - p_inf
    # we pulled p_ref towards zero in loss, so p_inf ~ 0 assumption for visualization.

    foil_t = torch.tensor(foil_xy.astype(np.float32), device=device)
    with torch.no_grad():
        uvp_foil = model(foil_t).cpu().numpy()
    p_surf = uvp_foil[:, 2]
    cp = p_surf - 0.0

    # Approximate Cl by integrating -Cp * n_y * ds (pressure-only) ---> rough
    # Compute normals approx using polygon geometry
    xpoly = foil_xy[:, 0]
    ypoly = foil_xy[:, 1]
    nx_poly = np.empty_like(xpoly)
    ny_poly = np.empty_like(ypoly)
    ds = np.empty_like(xpoly)

    for i in range(len(xpoly)):
        x0, y0 = xpoly[i], ypoly[i]
        x1, y1 = xpoly[(i + 1) % len(xpoly)], ypoly[(i + 1) % len(xpoly)]
        tx, ty = x1 - x0, y1 - y0
        ds[i] = math.hypot(tx, ty)
        nx_i, ny_i = ty, -tx
        nmag = math.hypot(nx_i, ny_i) + 1e-12
        nx_poly[i] = nx_i / nmag
        ny_poly[i] = ny_i / nmag

    # integrate Cp * normal_y * ds
    Cl_approx = -np.sum(cp.flatten() * ny_poly * ds) / 1.0
    print(f'Approx lift coefficient (pressure-only) Cl ~ {Cl_approx:.4f}')

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].contourf(X, Y, np.sqrt(U**2 + V**2), levels=50)
    ax[0].plot(foil_xy[:, 0], foil_xy[:, 1], 'k')
    ax[0].set_title('Velocity magnitude')
    ax[1].plot(p_surf, label='p on foil')
    ax[1].set_title('Pressure on foil (surface)')
    plt.show()

    return model, history


# Choose device
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', dev)
model, history = train_pinn(n_epochs=1200, device=dev, n_collocation=15000, show_every=200)

# Save Model
torch.save(model.state_dict(), 'pinn_airfoil_model.pt')
print('Model saved to pinn_airfoil_model.pt')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        