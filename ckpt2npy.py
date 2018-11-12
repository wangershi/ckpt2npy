from tensorflow.python import pywrap_tensorflow
import numpy as np

checkpoint_path = 'inception_resnet_v2_2016_08_30.ckpt'	# your ckpt path
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

inception_resnet_v2 = {}

for key in var_to_shape_map:
	sStr_2 = key
	if not sStr_2 in inception_resnet_v2.keys():
		inception_resnet_v2[sStr_2]=[reader.get_tensor(key)]
	else:
		raise Exception("Same key in the same network!!!")

np.save('inception_resnet_v2.npy', inception_resnet_v2)
