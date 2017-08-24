Preparation:

1. Create a folder ‘dataset’

2. Download dataset from https://drive.google.com/drive/folders/0ByVbpbF2lPbmVDl1aDZRQUpGOFk

3. Place downloaded folders in dataset. e.g.. (./dataset/cropped/predicting)

For Training:
- cd ./training
- run python main.py --do_training=True

For Predicting:
- cd ./training
- run python main.py

Optional arguments:

--do_training: Whether to do training or predicting, False to do predict. default: False

--dataset_dir: Directory of training dataset. default: ‘../dataset/cropped’

--logging_dir: Directory of logs (models). default: ‘../log’

--dataset_dir: Directory of predicting files. default: ‘../dataset/predicting/cropped’

--model_name: Predefined models: Simple, Custom, Resnet_v1, Resnet_v2, or cifar10. default: ‘Simple’

--image_format: Image format (png or jpg). default: ‘png’

--image_size: Image size (height, width) in pixels. default: 200

--margin: Margin for the crop around the bounding box (height, width) in pixels. default: 44

--max_step: Maximum training steps. default: 400
