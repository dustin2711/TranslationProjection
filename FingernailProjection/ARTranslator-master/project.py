import cv2 as cv
import numpy as np

# Initialize the camera capture object
cam = cv.VideoCapture(0)

# Load the calibration data from an XML file
fs = cv.FileStorage("calibration_result.xml", cv.FILE_STORAGE_READ)
cam_shape = fs.getNode("img_shape").mat()
ret = fs.getNode("rms")
cam_int = fs.getNode("cam_int").mat()
cam_dist = fs.getNode("cam_dist").mat()
proj_int = fs.getNode("proj_int").mat()
proj_dist = fs.getNode("proj_dist").mat()
cam_proj_rmat = fs.getNode("rotation").mat()
cam_proj_tvec = fs.getNode("translation").mat()

# Perform stereo rectification
R1, R2, P1, P2, Q, ROI1, ROI2 = cv.stereoRectify(
    cam_int,
    cam_dist,
    proj_int,
    proj_dist,
    (1280, 960),
    cam_proj_rmat,
    cam_proj_tvec,
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=0,
)

# Define resolution
w = 1280
h = 960
res = (int(w), int(h))
res2 = (int(w), int(h))

# Print the Q matrix
print(Q)

# Initialize undistort and rectify maps for the camera
a, b = cv.initUndistortRectifyMap(cam_int, cam_dist, R1, P1, res2, cv.CV_32FC1)

# Print a specific element from the rotation matrix
print(cam_proj_rmat[0][1])

# Create a transformation matrix T
T = np.array(
    [
        [cam_proj_rmat[0][0], cam_proj_rmat[0][1], cam_proj_tvec[0][0]],
        [cam_proj_rmat[1][0], cam_proj_rmat[1][1], cam_proj_tvec[1][0]],
        [0, 0, 1],
    ]
)

# Initialize virtual camera and mesh generator (commented out as they are not defined)
# c1 = vcam(H=h, W=w)
# c1.set_tvec(cam_proj_tvec[0][0], cam_proj_tvec[1][0], cam_proj_tvec[2][0])
# c1.set_rvec(0, 0, -5)
# c1.sx = 0.3
# c1.sy = 0.3
# plane = meshGen(h, w)
# plane.Z = plane.X * 0 + 1

# Initialize transformation parameters
a = 0
b = -1.5
g = -8
x = 50
y = -300
z = cam_proj_tvec[2][0]
l = 0.34

# Main loop for capturing and processing frames
while True:
    # Reinitialize virtual camera and mesh generator in each iteration (commented out as they are not defined)
    # c1 = vcam(H=h, W=w)
    # c1.set_tvec(x, y, z)
    # c1.set_rvec(a, b, g)
    # c1.sx = l
    # c1.sy = l
    # plane = meshGen(h, w)
    # plane.Z = plane.X * 0 + 1
    # pts3d = plane.getPlane()
    # pts2d = c1.project(pts3d)

    # Generate remap matrices (commented out as they are not defined)
    # map_x, map_y = c1.getMaps(pts2d)

    # Capture frame from camera
    ret, frame = cam.read()
    h, w = frame.shape[:2]

    # Apply remap to the frame (commented out as they are not defined)
    # test = cv.remap(frame, a, b, cv.INTER_LINEAR)
    # test = cv.remap(frame, map_x, map_y, interpolation=cv.INTER_LINEAR)

    # Apply perspective transformation (commented out as they are not defined)
    # test = cv.warpPerspective(test, T, res)

    # Apply undistort and rectify maps (commented out as they are not defined)
    # test = cv.remap(test, c, d, cv.INTER_LINEAR)
    # test = cv.undistort(frame, cam_int, cam_dist, None, cam_int)

    # Display original and processed frames
    cv.imshow("original", frame)
    # cv.imshow("test", test)

    # Wait for key press
    k = cv.waitKey(1) & 0xFF

    # Exit loop if 'q' is pressed
    if k == ord("q"):
        break
    # Adjust parameters based on key presses
    if k == ord("a"):
        a += 1
        print(a)
    if k == ord("b"):
        b += 0.1
        print(b)
    if k == ord("p"):
        b -= 0.1
        print(b)
    if k == ord("g"):
        g += 1
        print(g)
    if k == ord("x"):
        x += 1
        print(x)
    if k == ord("y"):
        y -= 1
        print(y)
    if k == ord("z"):
        z += 1
        print(z)
    if k == ord("l"):
        l += 0.01
        print(l)
