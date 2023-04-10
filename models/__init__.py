from models.Classifier import Classifier, NormalizedClassifier
from models.ResNet import ResNet50
from models.resnet1 import ResNet50_1
from models.resnet6 import ResNet50_6
from models.resnet2 import ResNet50_2
from models.resnet3 import ResNet50_3
from models.resnet4 import ResNet50_4
from models.resnet5 import ResNet50_5
from models.resnet7 import ResNet50_7
from models.resnet8 import ResNet50_8
from models.resnet9 import ResNet50_9
from models.resnet10 import ResNet50_10
from models.resnet11 import ResNet50_11
from models.resnet_ibn import resnet50_ibn_a
from models.cbam import ResNet50_cbam
from models.se import ResNet50_se
from models.cia import ResNet50_cia
from models.eca import ResNet50_eca
from models.psa import ResNet50_psa
from models.shuffle import ResNet50_shuffle

def build_model(config, num_classes):
	# Build backbone
	print("Initializing model: {}".format(config.MODEL.NAME))
	if config.MODEL.NAME == 'resnet50':
		model = ResNet50(res4_stride=config.MODEL.RES4_STRIDE)
		# model = resnet50_ibn_a(last_stride=config.MODEL.RES4_STRIDE)
	else:
		raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	# Build classifier
	if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
		classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)
		# clothclassifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
	else:
		classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)

	return model, classifier 