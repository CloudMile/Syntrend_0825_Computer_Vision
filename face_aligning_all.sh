
echo "--------------------"
echo "Cropping images..."
echo "  image size  = 200"
echo "  margin      = 32"
python3 ./tf_face_mtcnn/align/align_dataset_mtcnn.py ./dataset/downloaded ./dataset/cropped__ --image_size 200 --margin 32
