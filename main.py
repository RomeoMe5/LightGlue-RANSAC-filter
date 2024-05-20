import cv2
from lightglue import LightGlue, SuperPoint, ALIKED, DISK, SIFT, match_pair
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import math
from skimage.measure import ransac
from skimage.transform import AffineTransform
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

def get_map_pic_2(path_map):
  map = cv2.imread(path_map, cv2.IMREAD_COLOR)
  mx, my, mdx, mdy = 1000, 1000, 2100, 2100
  map = map[mx:mx+mdx, my:my+mdy]
  tsat_img = transforms.ToTensor()(map[:,:,0:4])
  return tsat_img

pip freeze > requirements.txt
def get_mean(m_kpts0):
  min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
  for x, y in m_kpts0:
      min_x = min(min_x, x)
      min_y = min(min_y, y)
      max_x = max(max_x, x)
      max_y = max(max_y, y)

  edge_points = [
      (min_x, max_y),
      (max_x, max_y),
      (max_x, min_y),
      (min_x, min_y)
  ]
  mean_coord_x = np.mean([int(min_x), int(max_x)])
  mean_coord_y = np.mean([int(min_y), int(max_y)])
  return [mean_coord_x+1000, mean_coord_y+1000]


def initialize_particles(num_particles, map_size):
  particles = np.random.rand(num_particles, 2) * map_size
  return particles


def update_particles(particles, movement_model):
  particles -= movement_model
  return particles


def calculate_angle(x1, y1, x2, y2):
  delta_x = x2 - x1
  delta_y = y2 - y1
  angle_rad = math.atan2(delta_y, delta_x)
  angle_deg = math.degrees(angle_rad)
  if angle_deg < 0:
      angle_deg += 360

  return angle_deg


def update_weights(particles, measurements, measurement_noise):
  weights = np.zeros(len(particles))
  for i, particle in enumerate(particles):
      distance_to_measurements = np.linalg.norm(measurements - particle, axis=1)
      prob_measurement = np.exp(-0.5 * (distance_to_measurements**2) / (measurement_noise**2))
      weights[i] = np.prod(prob_measurement)
  weights /= np.sum(weights)
  return weights


def resample_particles(particles, weights):
  indices = np.random.choice(len(particles), size=len(particles), replace=True, p=weights)
  return particles[indices]


def move_drone(x0, y0, angle, speed_per_pixel):
  angle_rad = math.radians(angle)
  delta_x = speed_per_pixel * math.cos(angle_rad)
  delta_y = speed_per_pixel * math.sin(angle_rad)

  new_x = x0 - delta_x
  new_y = y0 - delta_y
  return new_x, new_y

num_particles = 2000
map_size = 4800
measurement_noise = 20

initial_x = 3044
initial_y = 2147
init_xy = [initial_x, initial_y]

particles = initialize_particles(num_particles, map_size)
angles = []
final_coords = []

tsat_img = get_map_pic_2("/Maps/AdM-03-2014.png")

for t in range(1, 31):
  if 49 + t * 2 > 100:
    path_flight = f"/Flight_1/image000{49 + t * 2}.png"
  else:
    path_flight = f"/Flight_1/Flight 1/image0000{49 + t * 2}.png"

  uav = cv2.imread(path_flight, cv2.IMREAD_COLOR)
  uav = cv2.resize(uav, (0,0), fx=0.3, fy=0.3)
  uav = cv2.rotate(uav, cv2.ROTATE_90_COUNTERCLOCKWISE)
  tframe = transforms.ToTensor()(uav)

  image0 = tsat_img
  image1 = tframe
  feats0 = extractor.extract(image0.to(device))
  feats1 = extractor.extract(image1.to(device))
  matches01 = matcher({"image0": feats0, "image1": feats1})

  feats0, feats1, matches01 = [
      rbd(x) for x in [feats0, feats1, matches01]
  ]

  kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
  m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
  print(len(m_kpts0))

  source_points = np.array([[x, y] for x, y in m_kpts0.detach().cpu().numpy()])
  destination_points = np.array([[x, y] for x, y in m_kpts1.detach().cpu().numpy()])

  model, inliers = ransac((source_points, destination_points), AffineTransform, min_samples=3, residual_threshold=5, max_trials=1000)
  matrix = model.params
  inlier_indices = np.nonzero(inliers)[0]
  inlier_source_points = source_points[inlier_indices]
  inlier_destination_points = destination_points[inlier_indices]
  print(len(inlier_source_points))

  axes = viz2d.plot_images([image0, image1])
  inlier_source_points = inlier_source_points.reshape(-1, 2)
  inlier_destination_points = inlier_destination_points.reshape(-1, 2)
  viz2d.plot_matches(inlier_source_points, inlier_destination_points, color="lime", lw=0.2)
  viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
  plt.show()

  curr = get_mean(inlier_source_points)

  initial_angle = calculate_angle(curr[0], curr[1], initial_x, initial_y)
  angles.append(initial_angle)
  print(initial_angle)

  speed_per_sec_pix = np.linalg.norm(np.array(curr)-np.array(init_xy))
  new_x, new_y = move_drone(initial_x, initial_y, initial_angle, speed_per_sec_pix)

  angle_rad = math.radians(initial_angle)
  measurements = np.array([[new_x, new_y]])
  particles = update_particles(particles, np.array([speed_per_sec_pix*math.cos(angle_rad), speed_per_sec_pix*math.sin(angle_rad)]))
  weights = update_weights(particles, measurements, measurement_noise)
  particles = resample_particles(particles, weights)
  estimated_position = np.mean(particles, axis=0)

  print(f"Кадр {49+t*2}: Оценка позиции БПЛА: ({estimated_position[0]}, {estimated_position[1]})")

  final_coords.append((initial_x, initial_y))
  initial_x = estimated_position[0]
  initial_y = estimated_position[1]
  init_xy = [initial_x, initial_y]


data = pd.read_csv('/Flight_1/traj1_ground_truth.txt', delim_whitespace=True)
data.to_csv('/Flight_1/output_file.csv', index=False)
df = pd.read_csv("/Flight_1/output_file.csv")

data['coordinates'] = list(zip(data['px-x'], data['px-y']))

every_third_coordinate = data['coordinates'][::3].tolist()
every_third_coordinate = every_third_coordinate[:30]
print(every_third_coordinate)
print(angles)

distances = np.linalg.norm(np.array(final_coords) - np.array(every_third_coordinate), axis=1)
print(distances)

sizes = [49 + 2*i for i in range(1, 31)]
plt.xlabel('Номер кадра')
plt.ylabel('AED (pix)')
plt.plot(sizes, distances)
plt.show()