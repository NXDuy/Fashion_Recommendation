import os
from io import BytesIO
import requests
from requests.adapters import HTTPAdapter
from PIL import Image
import torchvision.transforms as transforms
import requests
import torch
from yolov5.utils.torch_utils import select_device
from annoy import AnnoyIndex
from yolov5.models.common import DetectMultiBackend
from tqdm import tqdm
from PIL import UnidentifiedImageError

from utils import batch_in_thread_pool

class Recommendation():
  def __init__(self, model_path: str="yolov5s-cls.pt", device=0):
    self.requests_session = requests.Session()
    adapter = HTTPAdapter(pool_connections=150, pool_maxsize=150)
    self.requests_session.mount('http://', adapter)
    self.requests_session.mount('https://', adapter)

    if isinstance(device, int):
      self.device = f"cuda:{device}"
    else:
      self.device = device
    self.model = DetectMultiBackend('yolov5s-cls.pt', device=select_device(device, batch_size=16))

  def get_products(self, database="dressup", user="netflox", password="netflox123", host="15.235.197.43", port="2432"):
    import psycopg2
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cursor = conn.cursor()
    cursor.execute("""
      SELECT id, image_urls FROM product WHERE is_public
    """
    )
    self.products = cursor.fetchall()
    conn.close()

  def extract_feature(self, paths: list[str], size=224):
    def _transform_image_by_path(path: str):
      if os.path.exists(path):
        im = Image.open(path)
      else:
        try:
          im = Image.open(BytesIO(self.requests_session.get(path).content))
        except UnidentifiedImageError:
          return
      if im.mode != "RGB":
        return
      im = im.resize((size, size))
      return to_tensor(im).unsqueeze(dim=0).to(self.device)

    features, ims = [], []
    to_tensor = transforms.ToTensor()
    ims = batch_in_thread_pool(func=_transform_image_by_path, items=paths)

    ims = torch.cat(ims) if len(ims) > 0 else ims[0].unsqueeze(dim=0)
    features = self.model(ims)
    features = torch.mean(features, dim=0) if len(features) > 1 else features.squeeze(dim=0)
    return features.cpu().detach().numpy()

  def build_tree(self, idx, feature):
    if not hasattr(self, "tree"):
      self.tree = AnnoyIndex(len(feature), 'angular')
    self.tree.add_item(idx, feature)
  
  def get_similar_images(self, img_index, number_of_items=12):
    similar_img_ids = self.tree.get_nns_by_item(img_index, number_of_items+1)
    # ignore first item as it is always target image
    product_ids = [self.products[id][0] for id in similar_img_ids]
    return product_ids

  def get_similar_images_centroid(self, vector_value, number_of_items=12):
      similar_img_ids = self.tree.get_nns_by_vector(vector_value, number_of_items+1)
      # ignore first item as it is always target image
      product_ids = [self.products[id][0] for id in similar_img_ids]
      return product_ids

  def get_index(self, product_id: str):
    for id, (product_id, paths) in enumerate(self.products):
      if product_id == product_id:
        return id

  def build(self, database="dressup", user="netflox", password="netflox123", host="15.235.197.43", port="2432"):
    self.get_products(database, user, password, host, port)
    for id, (product_id, paths) in tqdm(enumerate(self.products)):
      self.build_tree(id, self.extract_feature(paths))
    self.tree.build(100)
  
  def __call__(self, paths: list[str], num_recommendations=12):
    if paths is None or len(paths) == 0:
      return None
    user_feature = self.extract_feature(paths)
    return self.get_similar_images_centroid(user_feature, num_recommendations)

  
recommendation = Recommendation(device="cpu")
