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

def get_points(img, threshold, color):
  global points
  cur_len = len(points)
  while len(points) < threshold:
    if len(points) % 2 == 0 and len(points) != cur_len: 
      cv2.line(img, tuple(points[-1][:2]), tuple(points[-2][:2]), color, 2)
      cv2.imshow("img", img)
    key = cv2.waitKey(20)
    if key == ord('q'):
      quit()

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


def rectifyAffine(points):
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

  return H1

def rectifyMetric(points):
  l1 = np.cross(points[0], points[1])
  m1 = np.cross(points[2], points[3])
  l2 = np.cross(points[4], points[5])
  m2 = np.cross(points[6], points[7])

  RHS = np.zeros((2, 1))
  A = np.zeros((2, 2))

  RHS[0] = -l1[1] * m1[1]
  RHS[1] = -l2[1] * m2[1]

  A[0][0] = l1[0] * m1[0]
  A[0][1] = l1[0] * m1[1] + l1[1] * m1[0]
  A[1][0] = l2[0] * m2[0]
  A[1][1] = l2[0] * m2[1] + l2[1] * m2[0]

  s = np.linalg.lstsq(A, RHS)[0]
  S = np.array([[s[0][0], s[1][0]],[s[1][0],1]])

  u, s, vh = svd(S, full_matrices=1, compute_uv=1)
  A = np.linalg.cholesky(S)

  H2 = np.zeros((3, 3))
  H2[:2, :2] = A
  H2[2, 2] = 1

  H2_inv = np.linalg.inv(H2)

  return H2_inv

def rectify(img_path, pre_points):
  global points
  points += pre_loaded_points

  img = cv2.imread(img_path)
  img_clean = np.copy(img)
  img_shape = img.shape

  cv2.imshow("img", img)
  cv2.setMouseCallback("img", get_point)

  print("Affine Rectification:")
  print("Highlight 2 pairs of parallel lines")
  get_points(img, 8, (255, 0, 0))
  H_af = rectifyAffine(points)

  # Affinely rectified
  img, _, _ = perspective_warp(img_clean, H_af)
  img_clean = np.copy(img)

  cv2.imshow("img", img_clean)

  print("Metric Rectification:")
  print("Highlight 2 pairs of orthogonal lines")
  get_points(img, 16, (0, 0, 255))
  H_me = rectifyMetric(points[8:])

  # Metrically rectified:
  img, _, _ = perspective_warp(img_clean, H_me)

  cv2.imshow("img", img)

  key = cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  pre_loaded_points = []
  if len(sys.argv) == 2:
    pass
  elif len(sys.argv) == 3:
    point_path = sys.argv[2]
    pre_points = np.genfromtxt(point_path, delimiter=",").astype(int)
    pre_loaded_points = [[p[0], p[1], p[2]] for p in pre_points]
  else:
    print("Usage: python rectify.py image_path")
  path = sys.argv[1] 
  rectify(path, pre_loaded_points)
