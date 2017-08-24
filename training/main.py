import train as tra
import preparation as prep
import predict as pred
import sys
import os
import argparse

def main(args):
	print(args)
	if (args.do_training):
		tra.run_training(args.dataset_dir, args.logging_dir,
			model_name=args.model_name,
			img_format=args.image_format, 
			max_step=args.max_step)
	else:
		images = prep.readImagesInDirectory(args.predict_dir, format=args.image_format)
		model_dir = args.logging_dir + '/' + args.model_name + '/train'
		result, labels = pred.classify_images(args.model_name, model_dir, images, labels=['Hsia Yu-chiao', 'Sung Yun-hua'])
		print(labels)
		print(result)

def parse_arguments(argv):
    parser = argparse.ArgumentParser() 
    parser.add_argument('--do_training', type=bool, 
    	help='Whether to do training or predicting, False to do predict. default False', default=False)
    parser.add_argument('--dataset_dir', type=str, 
    	help='Directory of training dataset', default='../dataset/cropped')
    parser.add_argument('--logging_dir', type=str, 
    	help='Directory of logs (models)', default='../log')
    parser.add_argument('--predict_dir', type=str, 
    	help='Directory of predicting files', default='../dataset/predicting/cropped')
    parser.add_argument('--model_name', type=str, 
    	help='Predefined models: Simple, Custom, Resnet_v1, Resnet_v2, or cifar10', default='Simple')
    parser.add_argument('--image_format', type=str,
        help='Image format (png or jpg)', default='png')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=200)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--max_step', type=int,
        help='Maximum training steps', default=400)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
