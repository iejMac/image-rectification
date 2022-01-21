import cv2
import sys
import math
import numpy as np

from scipy.linalg import null_space
from numpy.linalg import cholesky, inv, svd

# TEMP GLOBALS:
points = []

def get_point(event, x, y, flags, params):
  global points
  if event == cv2.EVENT_LBUTTONDOWN:
    points.append([x, y, 1])
    print(x, y)

def perspective_warp(image, transform):
	h, w = image.shape[:2]
	corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
	corners_aft = cv2.perspectiveTransform(corners_bef, transform)
	xmin = math.floor(corners_aft[:, 0, 0].min())
	ymin = math.floor(corners_aft[:, 0, 1].min())
	xmax = math.ceil(corners_aft[:, 0, 0].max())
	ymax = math.ceil(corners_aft[:, 0, 1].max())
	x_adj = math.floor(xmin - corners_aft[0, 0, 0])
	y_adj = math.floor(ymin - corners_aft[0, 0, 1])
	translate = np.eye(3)
	translate[0, 2] = -xmin
	translate[1, 2] = -ymin
	corrected_transform = np.matmul(translate, transform)
	return cv2.warpPerspective(image, corrected_transform, (math.ceil(xmax - xmin), math.ceil(ymax - ymin))), x_adj, y_adj

def rectify(img_path):
  global points

  img = cv2.imread(img_path)
  img_shape = img.shape

  cv2.imshow("test", img)
  cv2.setMouseCallback("test", get_point)

  while len(points) < 16:
    if len(points) % 2 == 0 and len(points) != 0: 
      if len(points) < 9:
        cv2.line(img, tuple(points[-1][:2]), tuple(points[-2][:2]), (255, 0, 0), 2)
      else:
        cv2.line(img, tuple(points[-1][:2]), tuple(points[-2][:2]), (0, 0, 255), 2)
      cv2.imshow("test", img)
    
    key = cv2.waitKey(20)
    if key == ord('q'):
      quit()

  # Affine rectification:
  l1 = np.cross(points[0], points[1])
  l2 = np.cross(points[2], points[3])
  m1 = np.cross(points[4], points[5])
  m2 = np.cross(points[6], points[7])

  p1 = np.cross(l1, l2)
  p2 = np.cross(m1, m2)

  p1 = (p1 / p1[2]).astype(int)
  p2 = (p2 / p2[2]).astype(int)

  img_l_inf = np.cross(p1, p2).astype(float)
  img_l_inf /= np.linalg.norm(img_l_inf)
  img_l_inf /= img_l_inf[2]

  H1 = np.eye(3)
  H1[2] = img_l_inf 

  # Metric rectification:
  l1 = np.cross(points[8], points[9])
  m1 = np.cross(points[10], points[11])
  l2 = np.cross(points[12], points[13])
  m2 = np.cross(points[14], points[15])

  eq = np.zeros((2, 3))
  eq[0] = np.array([l1[0] * m1[0], l1[0]*m1[1] + l1[1]*m1[0], l1[1]*m1[1]])
  eq[1] = np.array([l2[0] * m2[0], l2[0]*m2[1] + l2[1]*m2[0], l2[1]*m2[1]])

  S = null_space(eq).T[0]
  S = np.append(S, 1.0).reshape((2, 2))
  
  U, D, V = svd(S)

  A = U * np.sqrt(D) * U.T
  K = cholesky(S)

  H2 = np.zeros((3, 3))
  H2[0:2, 0:2] = A
  H2[2][2] = 1.0

  H2_inv = inv(H2)

  img = cv2.imread(img_path)
  dst, _, _ = perspective_warp(img, H1)
  dst, _, _ = perspective_warp(dst, H2_inv)

  cv2.imshow("test", dst)


  key = cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python rectify.py image_path")
  rectify(sys.argv[1])
