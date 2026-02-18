#!/usr/bin/env python
import yaml
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.ndimage import maximum_filter1d

# ================= PARAMETERS =================
SAFETY_MARGIN_METERS = 0.2


# Phase 1: Laplacian smoothing (nettoie le squelette brut)
SMOOTH_ITERATIONS = 150
SMOOTHING_FACTOR = 0.9

# Phase 2: Minimisation de courbure
CURV_ITERATIONS = 400
CURV_WEIGHT = 1.5        # force de courbure (large car peu de points)
CURV_SMOOTH = 0.08       # anti-oscillation léger
CURV_DECAY = 0.998       # décroissance du pas
N_CONTROL = 80           # points de contrôle (clé du fix)
CURVATURE_STENCIL = 4    # regarde K pas devant/derrière

# Debug mode
DEBUG = True
DEBUG_DIR = "debug_output"

yaml_path=r'/home/alek/catkin_ws/src/f1tenth_simulator/maps/map_circuit_13_02.yaml'
# ==============================================




def save_debug_image(name, img, cmap='gray'):
    """Save debug visualization if DEBUG mode is on."""
    if not DEBUG:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    if len(img.shape) == 2:
        # Normalize for visibility
        if img.max() <= 1:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = img.astype(np.uint8)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"), img_save)
    else:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"), img)
    
    print(f"  [DEBUG] Saved {name}.png")


def points_to_image(path_px,color=(0, 0, 255), thickness=1):
    """
    Overlay a path on the map image 
    
    """
    global yaml_path
    map_img=load_map_from_yaml(yaml_path)[0]
    overlay = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    

    # Draw the path as a polyline
    
    path=np.array(path_px).astype(np.int32).reshape(-1,1,2)
    
    cv2.polylines(overlay, path, isClosed=True, color=color, thickness=thickness)

    return overlay

def load_map_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)
    map_dir = os.path.dirname(os.path.abspath(yaml_path))
    image_path = os.path.join(map_dir, map_data['image'])
    resolution = map_data['resolution']
    origin = map_data['origin']
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img, resolution, origin




def resample_uniform(path, num_points):
    """
    Redistributes the points with uniform spacing along the path
    """
    # Fermer la boucle pour l'interpolation
    closed = np.vstack([path, path[0:1]])
    diffs = np.diff(closed, axis=0)
    seg_len = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum_len[-1]

    targets = np.linspace(0, total, num_points, endpoint=False)
    new_path = np.zeros((num_points, 2))
    new_path[:, 0] = np.interp(targets, cum_len, closed[:, 0])
    new_path[:, 1] = np.interp(targets, cum_len, closed[:, 1])
    return new_path


# Iteratively remove "endpoint" pixels (pixels with only
#          1 skeleton neighbor). Branches shrink from their tips
#          inward until only the main loop remains (every pixel
#          has exactly 2 neighbors).

def prune_skeleton(skeleton, min_points=50):
    skel = skeleton.astype(np.uint8).copy()
    # Kernel counts 8-connected neighbors (excludes center)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    for _ in range(1000):
        neighbor_count = cv2.filter2D(skel, -1, kernel)
        endpoints = (skel == 1) & (neighbor_count <= 1)

        if not np.any(endpoints):
            break  # No more endpoints → only loops remain

        skel[endpoints] = 0

        if np.sum(skel) < min_points:
            print("WARNING: Pruning too aggressive, stopping early.")
            break
    
    return skel > 0



# Trace pixels using strict 8-connectivity. We only ever
#          move to an immediately adjacent pixel, so jumping across
#          walls is impossible. Direction continuity picks the best
#          neighbor when junctions exist.

def trace_skeleton_loop(skeleton_img):
    h, w = skeleton_img.shape
    visited = np.zeros((h, w), dtype=bool)

    ys, xs = np.where(skeleton_img > 0)
    if len(xs) == 0:
        raise ValueError("Empty skeleton after pruning!")

    start_x, start_y = int(xs[0]), int(ys[0])
    ordered = [(start_x, start_y)]
    visited[start_y, start_x] = True

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    prev = None
    current = (start_x, start_y)

    while True:
        cx, cy = current
        neighbors = []
        for dx, dy in offsets:
            nx, ny = cx + dx, cy + dy
            if 0 <= ny < h and 0 <= nx < w:
                if skeleton_img[ny, nx] > 0 and not visited[ny, nx]:
                    neighbors.append((nx, ny))

        if not neighbors:
            break

        # Direction continuity: prefer neighbor aligned with travel direction
        if prev is not None and len(neighbors) > 1:
            dir_x = current[0] - prev[0]
            dir_y = current[1] - prev[1]
            best = max(neighbors,
                       key=lambda n: (n[0] - cx) * dir_x + (n[1] - cy) * dir_y)
        else:
            best = neighbors[0]

        prev = current
        current = best
        visited[current[1], current[0]] = True
        ordered.append(current)
    save_debug_image("06_skeleton_loop",points_to_image(ordered))
    return np.array(ordered, dtype=np.float64)


def optimize_path(path, dist_transform, margin_px):
    """
    Phase 1 – Lissage laplacien sur le chemin dense (~300 pts)
    Phase 2 – Minimisation de courbure sur N_CONTROL points
              espacés, avec stencil large et rééchantillonnage.
    """
    path = path.astype(np.float64).copy()
    N = len(path)
    h, w = dist_transform.shape

    def is_safe(px, py):
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < h and 0 <= ix < w:
            return dist_transform[iy, ix] >= margin_px
        return False

    # ── Phase 1 : lissage laplacien sur chemin dense ──────────
    print("  Phase 1 : lissage laplacien …")
    for _ in range(SMOOTH_ITERATIONS):
        for i in range(N):
            p_prev = path[(i - 1) % N]
            p_next = path[(i + 1) % N]
            proposed = path[i] + SMOOTHING_FACTOR * (
                (p_prev + p_next) / 2.0 - path[i])
            if is_safe(proposed[0], proposed[1]):
                path[i] = proposed

    save_debug_image("07a_after_smoothing", points_to_image(path))

    # ── Rééchantillonner à N_CONTROL points espacés ───────────
    path = resample_uniform(path, N_CONTROL)
    N = N_CONTROL
    print(f"  Rééchantillonné à {N} points de contrôle")

    # ── Phase 2 : minimisation de courbure ────────────────────
    print("  Phase 2 : minimisation de courbure …")
    K = CURVATURE_STENCIL
    alpha_c = CURV_WEIGHT

    for iteration in range(CURV_ITERATIONS):
        new_path = path.copy()
        total_kappa_sq = 0.0

        for i in range(N):
            # Stencil large : voisins à K pas (pas 1 pas)
            p0 = path[(i - K) % N]
            p1 = path[i]
            p2 = path[(i + K) % N]

            d1 = p1 - p0
            d2 = p2 - p1
            l1 = np.linalg.norm(d1) + 1e-8
            l2 = np.linalg.norm(d2) + 1e-8
            l12 = np.linalg.norm(p2 - p0) + 1e-8

            # Courbure de Menger
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            kappa = 2.0 * cross / (l1 * l2 * l12)
            total_kappa_sq += kappa * kappa

            # Tangente et normale (calculées sur le stencil large)
            tangent = (p2 - p0) / l12
            normal = np.array([-tangent[1], tangent[0]])

            # Force de courbure : pousse vers l'extérieur du virage
            f_curv = -kappa * normal * alpha_c

            # Lissage léger (voisins immédiats)
            mid = (path[(i - 1) % N] + path[(i + 1) % N]) / 2.0
            f_smooth = (mid - p1) * CURV_SMOOTH

            proposed = p1 + f_curv + f_smooth
            if is_safe(proposed[0], proposed[1]):
                new_path[i] = proposed

        path = new_path

        # Rééchantillonnage périodique : empêche l'agglutination
        if iteration % 25 == 24:
            path = resample_uniform(path, N_CONTROL)

        alpha_c *= CURV_DECAY

        if iteration % 80 == 0:
            print(f"    iter {iteration:4d}  Σκ² = {total_kappa_sq:.4f}  "
                  f"α = {alpha_c:.4f}")

    print(f"    final      Σκ² = {total_kappa_sq:.4f}")
    save_debug_image("07b_after_curvature_opt", points_to_image(path))
    return path

# After spline fitting, detect any point that violates the
#          safety margin and push it away from the nearest wall
#          using the gradient of the distance transform.

def push_from_walls(path, dist_transform, margin_px, iterations=50):
    path = path.copy()
    h, w = dist_transform.shape

    grad_x = cv2.Sobel(dist_transform, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(dist_transform, cv2.CV_64F, 0, 1, ksize=5)

    for _ in range(iterations):
        moved = 0
        for i in range(len(path)):
            cx = int(round(path[i, 0]))
            cy = int(round(path[i, 1]))
            if 0 <= cy < h and 0 <= cx < w:
                dist = dist_transform[cy, cx]
                if dist < margin_px:
                    gx = grad_x[cy, cx]
                    gy = grad_y[cy, cx]
                    norm = np.sqrt(gx**2 + gy**2) + 1e-8
                    push = (margin_px - dist) * 0.3
                    path[i, 0] += push * gx / norm
                    path[i, 1] += push * gy / norm
                    moved += 1
        if moved == 0:
            break
    
    return path


def pixels_to_meters(path_px, resolution, origin, img_height):
    path_m = np.zeros_like(path_px, dtype=np.float64)
    path_m[:, 0] = path_px[:, 0] * resolution + origin[0]
    path_m[:, 1] = (img_height - path_px[:, 1]) * resolution + origin[1]
    return path_m


def compute_velocity_profile(path_m):
    dx = np.gradient(path_m[:, 0])
    dy = np.gradient(path_m[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (np.power(dx**2 + dy**2, 1.5) + 1e-9)

    ds = np.sqrt(np.diff(path_m[:, 0])**2 + np.diff(path_m[:, 1])**2)
    ds = np.concatenate([[0], ds])

    kappa = maximum_filter1d(curvature, size=15, mode='wrap')
    v_max = np.sqrt(8.0 / (kappa + 0.001))
    v_max = np.clip(v_max, 1.0, 10.0)

    # Forward pass (acceleration limit)
    v_fwd = np.zeros_like(v_max)
    v_fwd[0] = v_max[0]
    for i in range(1, len(v_fwd)):
        v_fwd[i] = min(v_max[i], np.sqrt(v_fwd[i-1]**2 + 2 * 3.0 * ds[i]))

    # Backward pass (braking limit)
    v_bwd = np.zeros_like(v_max)
    v_bwd[-1] = v_fwd[-1]
    for i in range(len(v_bwd) - 2, -1, -1):
        v_bwd[i] = min(v_fwd[i], np.sqrt(v_bwd[i+1]**2 + 2 * 6.0 * ds[i+1]))

    return v_bwd


def main(yaml_file):
    print(f"Processing {yaml_file}...")

    img, res, origin = load_map_from_yaml(yaml_file)
    height, width = img.shape
    margin_px = SAFETY_MARGIN_METERS / res
    print(f"Map: {width}x{height} | {res} m/px | margin: {margin_px:.1f} px")

    # Binary free space
    _, binary = cv2.threshold(img, 230, 1, cv2.THRESH_BINARY)
    save_debug_image('01_threshold',binary)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)# Eliminate small blobs
    save_debug_image('02_morph_open',binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)#Eliminate small holes
    save_debug_image('03_morph_close',binary)

    # Skeleton → prune branches → trace
    skeleton = skeletonize(binary).astype(np.uint8)
    save_debug_image('04_skeleton',skeleton)
    pruned = prune_skeleton(skeleton)
    save_debug_image('05_pruned',pruned)
    print(f"Skeleton: {np.sum(skeleton)} px → Pruned: {np.sum(pruned)} px")

    dist_transform = cv2.distanceTransform(
        (binary * 255).astype(np.uint8), cv2.DIST_L2, 5)
    save_debug_image('##_distance_transform',dist_transform)
    raw_path = trace_skeleton_loop(pruned)
    
    print(f"Traced path: {len(raw_path)} points")

    # Downsample to ~300 control points
    step = max(1, len(raw_path) // 300)
    path_ds = raw_path[::step]

    # Optimize
    opt_path = optimize_path(path_ds, dist_transform, margin_px)

    # Spline (periodic only if endpoints are close)
    loop_gap = np.linalg.norm(opt_path[0] - opt_path[-1])
    is_loop = loop_gap < 50
    print(f"Loop: {is_loop} (gap: {loop_gap:.1f} px)")

    try:
        tck, u = splprep(opt_path.T, s=1.0, per=int(is_loop))
        u_new = np.linspace(0, 1, 1000)
        sx, sy = splev(u_new, tck)
        smooth_px = np.column_stack((sx, sy))
    except Exception as e:
        print(f"Spline failed ({e}), using optimized path")
        smooth_px = opt_path

    # Push spline points away from walls
    #smooth_px = push_from_walls(smooth_px, dist_transform, margin_px)

    # Convert + velocity
    final_path_m = pixels_to_meters(smooth_px, res, origin, height)
    velocities = compute_velocity_profile(final_path_m)

    # Save
    output_file = f"{os.path.basename(yaml_file).split('.')[0]}_raceline.csv"
    np.savetxt(output_file,
               np.column_stack((final_path_m, velocities)),
               delimiter=",", header="x_m,y_m,v_ref_m_s", comments="")
    print(f"Saved {len(final_path_m)} waypoints → {output_file}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(img, cmap='gray', origin='upper')
    axes[0].plot(smooth_px[:, 0], smooth_px[:, 1], 'r-', lw=2, label='Raceline')
    pys, pxs = np.where(pruned > 0)
    axes[0].plot(pxs, pys, 'b.', ms=0.3, alpha=0.3, label='Pruned skeleton')
    axes[0].legend()
    axes[0].set_title("Path on Map")

    axes[1].plot(velocities, 'g-')
    axes[1].set_title("Velocity Profile")
    axes[1].set_xlabel("Waypoint")
    axes[1].set_ylabel("m/s")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    main(yaml_path)