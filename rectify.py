import cv2
import sys
import math
import numpy as np

# TEMP GLOBALS:
lines = []

def get_point(event, x, y, flags, params):
  global lines
  if event == cv2.EVENT_LBUTTONDOWN:
    lines.append([x, y, 1])
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
  global lines

  img = cv2.imread(img_path)
  img_shape = img.shape

  while len(lines) < 8:
    cv2.imshow("test", img)
    cv2.setMouseCallback("test", get_point)
    key = cv2.waitKey(20)
    if key == ord('q'):
      quit()

  l1 = np.cross(lines[0], lines[1])
  l2 = np.cross(lines[2], lines[3])
  m1 = np.cross(lines[4], lines[5])
  m2 = np.cross(lines[6], lines[7])

  p1 = np.cross(l1, l2)
  p2 = np.cross(m1, m2)

  p1 = (p1 / p1[2]).astype(int)
  p2 = (p2 / p2[2]).astype(int)

  img_l_inf = np.cross(p1, p2).astype(float)
  img_l_inf /= np.linalg.norm(img_l_inf)
  img_l_inf /= img_l_inf[2]

  H = np.eye(3)
  H[2] = img_l_inf 

  dst, _, _ = perspective_warp(img, H)

  cv2.imshow("test", dst)


  key = cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python rectify.py image_path")
  rectify(sys.argv[1])
