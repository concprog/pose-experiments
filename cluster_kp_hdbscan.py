import argparse
import cv2
import numpy as np
from numba import njit, jit

# from numba import np
import fast_hdbscan
from matplotlib import colormaps, pyplot


def get_mpl_colormap(cmap_name):
    cmap = colormaps.get(cmap_name)

    # Initialize the matplotlib color map
    sm = pyplot.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 1, 3)


@njit(fastmath=True)
def to_gray(img):
    # Define weights as a constant array
    weights = np.array([0.2126, 0.7152, 0.0722])

    # Extract channels
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # Apply weights and sum
    gray = r * weights[0] + g * weights[1] + b * weights[2]

    return gray


@njit(fastmath=True)
def minmax_normalize(img):
    mn = np.min(img)
    mx = np.max(img)
    return (img - mn) / (mx - mn + 1e-8)


@njit(fastmath=True, cache=True, nogil=True)
def integral_numba(img):
    h, w = img.shape
    intg = np.empty((h + 1, w + 1), dtype=np.float32)
    intg[0, :] = 0.0
    intg[:, 0] = 0.0
    for y in range(1, h + 1):
        row_sum = 0.0
        for x in range(1, w + 1):
            row_sum += img[y - 1, x - 1]
            intg[y, x] = intg[y - 1, x] + row_sum
    return intg


def get_atmosphere(image, scatterlight):
    for _ in range(1):
        scatter_est = np.sum(scatterlight, axis=2)
        n_pixels = scatter_est.size
        n_search_pixels = int(n_pixels * 0.001)
        image_vec = image.reshape(n_pixels, 3)
        indices = np.argsort(scatter_est.flatten())[::-1]
        atmosphere = np.mean(image_vec[indices[:n_search_pixels], :], axis=0)

        atmosphere = np.repeat(
            atmosphere[np.newaxis, np.newaxis, :], scatter_est.shape[0], axis=0
        )
        atmosphere = np.repeat(atmosphere, scatter_est.shape[1], axis=1)

        sek = scatter_est.flatten()[indices[n_search_pixels]]

        mask = scatter_est <= sek
        scatterlight = scatterlight * np.repeat(mask[:, :, np.newaxis], 3, axis=2) + (
            2 / 3 * sek - scatterlight
        ) * np.repeat((~mask)[:, :, np.newaxis], 3, axis=2)

    return atmosphere, scatterlight


def underwater_mask_stuff(img, omega=0.8):
    imgvec = img.reshape(-1, 3)
    x_RGB = np.mean(imgvec, axis=0)
    x_mean = np.repeat(x_RGB[np.newaxis, np.newaxis, :], img.shape[0], axis=0)
    x_mean = np.repeat(x_mean, img.shape[1], axis=1)

    scat_basis = x_mean / np.maximum(
        np.sqrt(np.sum(x_mean**2, axis=2, keepdims=True)), 0.001
    )
    fog_basis = img / np.maximum(np.sqrt(np.sum(img**2, axis=2, keepdims=True)), 0.001)
    cs_sim = np.sum(scat_basis * fog_basis, axis=2, keepdims=True)

    scattering_light = (
        cs_sim
        * (
            np.sum(img, axis=2, keepdims=True)
            / np.maximum(np.sum(x_mean, axis=2, keepdims=True), 0.001)
        )
        * x_mean
    )
    A, scattering_light = get_atmosphere(img, scattering_light)
    T = 1 - omega * scattering_light
    T_m = T**2

    gaussian1 = cv2.GaussianBlur(T_m, (0, 0), sigmaX=1)
    gaussian2 = cv2.GaussianBlur(T_m, (0, 0), sigmaX=21)
    dog = gaussian1 - gaussian2

    dog = minmax_normalize(dog)
    dog = dog**2 + (T_m - gaussian1)
    dog = to_gray(dog)

    # fog_removed = (img - atmosphere) / np.maximum(T_ini, 0.001) + atmosphere
    return np.clip(T_m, 0, 1)


def sort_points(pts):
    centroid = np.mean(pts, axis=0)
    diff = pts - centroid
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    sorted_indices = np.argsort(-angles)
    sorted_pts = pts[sorted_indices]
    return sorted_pts


def extract_keypoints(keypoints, depth_image=None):
    print(depth_image.shape)
    if depth_image is not None:
        # Extract 3D points: (x, y, depth_value)
        return np.array(
            list(
                map(
                    lambda kp: [
                        kp.pt[0],
                        kp.pt[1],
                        depth_image[int(kp.pt[1])][int(kp.pt[0])],
                    ],
                    keypoints,
                )
            ),
            dtype=np.float32,
        )

    return np.array(list(map(lambda kp: kp.pt, keypoints)), dtype=np.float32)


def cluster(pts, min_cluster_size=10, min_samples=2):
    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit_predict(pts)
    objects = {
        label: pts[np.where(clusterer.labels_ == label)]
        for label in np.unique(clusterer.labels_)
    }
    return objects


def process_image(
    img,
    detector,
    draw_flag=False,
    colormap=None,
):
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t_mask = to_gray(underwater_mask_stuff(img_rgb / 255.0) * 255)

    if colormap is not None:
        t_mask_vis = cv2.applyColorMap((t_mask).astype(np.uint8), colormap)
    else:
        t_mask_vis = t_mask.copy()
    cv2.imshow("mask", t_mask_vis)
    keypoints, _desc = detector.detectAndCompute(img, None)
    pts = extract_keypoints(keypoints, t_mask)
    four_pts = pts
    objects = cluster(pts, min_cluster_size=10, min_samples=2)
    # color = {object: tuple(np.random.randint(0, 255, 3).tolist()) for object in objects}
    colors = []
    for i in range(20):
        idx = int(i * 255 / 20)
        color = colormap[idx]
        colors.extend(color.tolist())
    colors[1], colors[-1] = colors[-1], colors[1]
    # colors[2], colors[-2] = colors[-2], colors[2]
    colors[3], colors[-6] = colors[-6], colors[3]

    if four_pts is not None:
        for object_ in objects:
            if object_ == -1:
                continue
            # print((ag+object*ag, ab+object*ab, ar+object*ar))
            for pt in objects[object_]:
                pt_int = tuple(np.round(pt).astype(int))
                cv2.circle(img, pt_int[:-1], 5, colors[object_], -1)

    if four_pts is None or len(four_pts) != 4:
        return img, None, None

    try:
        sorted_pts = sort_points(four_pts)
    except:
        return img, None, None

    object_pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )

    fx = w
    fy = h
    cx = w / 2
    cy = h / 2
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        object_pts,
        sorted_pts.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        return img, None, None

    if draw_flag:
        axis_points = np.array(
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype=np.float32
        )
        imgpts, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin = tuple(imgpts[0])
        x_end = tuple(imgpts[1])
        y_end = tuple(imgpts[2])
        z_end = tuple(imgpts[3])
        cv2.line(img, origin, x_end, (0, 0, 255), 2)
        cv2.line(img, origin, y_end, (0, 255, 0), 2)
        cv2.line(img, origin, z_end, (255, 0, 0), 2)

    return img, rvec, tvec


def main():
    parser = argparse.ArgumentParser(
        description="Estimate object pose from video and save results"
    )
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--live", action="store_true", help="Use webcam as input")
    parser.add_argument(
        "--draw", action="store_true", help="Draw pose axes on the output"
    )
    parser.add_argument("--output", type=str, help="Path to save output video file")
    args = parser.parse_args()

    if not args.video and not args.live:
        print("Error: Must specify either --video or --live")
        return

    cmap = get_mpl_colormap("plasma")
    detector = cv2.AKAZE_create()

    if args.live:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    if not args.live:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not args.live else 30
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, rvec, tvec = process_image(frame, detector, args.draw, cmap)
        if rvec is not None and tvec is not None:
            print(f"Rotation: {rvec.ravel()}, Translation: {tvec.ravel()}")
        if args.live:
            cv2.imshow("Processed Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            out.write(processed_frame)

    cap.release()
    if not args.live:
        out.release()
    print(f"Saved processed video to: {args.output}")


if __name__ == "__main__":
    main()
