import os

def main():
	os.system("python convert_dataset_to_npy.py --config_file datasets/configs/tcia_slices.yaml --data_dir /local/DEEPLEARNING/TCIA/ --image_output_dir /local/DEEPLEARNING/pancreas_multi_resolution/images/ --annotation_output_dir /local/DEEPLEARNING/pancreas_multi_resolution/labels/ --save_mode volumes")
	#os.system("python convert_dataset_to_npy.py --config_file datasets/configs/tcia_slices.yaml --data_dir ~/datasets/ --image_output_dir ~/datasets/first_for_labels/images/ --annotation_output_dir ~/datasets/first_for_labels/labels/ --save_mode volumes")





if __name__== "__main__":
	main()