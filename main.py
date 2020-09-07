import os

def main():
	os.system("python convert_dataset_to_npy.py --config_file datasets/configs/tcia_slices.yaml --data_dir /local/HD_DEEPLEARNING/TCIA_pancreas/ --image_output_dir ~/data/TCIA_torch/images/ --annotation_output_dir ~/data/TCIA_torch/labels/ --save_mode volumes")
	#os.system("python convert_dataset_to_npy.py --config_file datasets/configs/tcia_slices.yaml --data_dir ~/datasets/ --image_output_dir ~/datasets/first_for_labels/images/ --annotation_output_dir ~/datasets/first_for_labels/labels/ --save_mode volumes")





if __name__== "__main__":
	main()