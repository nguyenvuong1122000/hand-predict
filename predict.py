from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.tool.predictor import Predictor
from PIL import Image
import numpy as np
config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = '/home/vuong/PycharmProjects/hand-predict/model/weight.pth'
config['cnn']['pretrained'] = True
config['device'] = 'cpu'
config['predictor']['beamsearch']=False

predictor = Predictor(config)
def predict(imgPath):
    img = Image.open(imgPath)
    return predictor.predict(img)