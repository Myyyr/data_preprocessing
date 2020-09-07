
class Pancreas3DPriorDataManager(DataManager):

    def __init__(self, data_dir, random_seed, prior_map, validation_data_proportion=0.2, gamma=1.0, use_mlp_position=False, **_ignored):
        super().__init__(data_dir, random_seed, validation_data_proportion)

        self.nb_classes = 2
        self.input_size = (512, 512, 1)
        #self.mean_value = 48.95255
        self.mean_value = 18.6461
        #self.std_value = 6.6069
        self.std_value = 21.5892

        if use_mlp_position:
            print('Attention : using MLP position')
            self.info_filename = "Info_pancreas_position_MLP_prediction.yaml"
        else:
            self.info_filename = "Info_pancreas_position.yaml"
            
        self.gamma = gamma
        self.delta = 10
        
        self._prior_map = np.load(prior_map)

        print('[WARNING] GAMMA = {}'.format(self.gamma))
        time.sleep(2)

        
        self.output_types = ({
            'image' : tf.float32,
            'prior_map' : tf.float32,
            'patient_id' : tf.int32,
            'slice_num' : tf.int32
        }, tf.uint8)

        self.output_shapes = ({
            'image' : tf.TensorShape([None, None, 3]),
            'prior_map' : tf.TensorShape([None, None, self.nb_classes]),
            'patient_id' : tf.TensorShape([]),
            'slice_num' : tf.TensorShape([]),
        }, tf.TensorShape([None, None, self.nb_classes]))

        
    def get_voxel_spacing(self, pid):
        annotation_path = os.path.join(self.data_dir, 'TCIA_pancreas_labels/label{}.nii')
        image_data, image_header = medload(annotation_path.format(pid))
        voxel_spacing = header.get_voxel_spacing(image_header)
        voxel_spacing = (voxel_spacing[2], voxel_spacing[0], voxel_spacing[1])
        return voxel_spacing

        
        
    def gen(self, examples_names):
        for i, filename in enumerate(examples_names):
            patient_id, slice_num = filename.split('.')[0].split('/')
            slice_num = int(slice_num)
            image = np.load(os.path.join(self.image_path, filename))
            
            #image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32)
            #image = image - self.mean_value
            image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
            
            annotation = np.load(os.path.join(self.annotation_path, filename))[:,:,1]

            annotation = tf.keras.utils.to_categorical(annotation, num_classes=2)

            if self._prior_map is None:
                prior_map = np.ones((self.input_size[0], self.input_size[1], self.nb_classes))
            else:
                # Get z_min and z_max for the pancreas annotation
                with open(os.path.join(self.data_dir, 'annotations', patient_id, self.info_filename), 'r') as f:
                    z_positions = yaml.load(f, Loader=yaml.FullLoader)

                z_positions['z_min'] -= self.delta
                z_positions['z_max'] += self.delta

                z_positions['z_min'] = 0
                z_positions['z_max'] = len(os.listdir('/local/DEEPLEARNING/TCIA_pancreas/annotations/'+patient_id)) - 2
                
                
                prior_map = np.zeros((self.input_size[0], self.input_size[1], self.nb_classes))
                if slice_num >= z_positions['z_min'] and slice_num <= z_positions['z_max']:
                    idx_in_prior = round((slice_num-z_positions['z_min']) * (self._prior_map.shape[-1]-1)  / (z_positions['z_max']-z_positions['z_min']))
                    for c in range(self.nb_classes):
                        prior_map[:,:,c] = self._prior_map[:,:,c,idx_in_prior]
                else:
                    prior_map[:,:,0] = np.ones((self.input_size[0], self.input_size[1]))


            # Prior map post-processing
            prior_map = prior_post_processing(prior_map, self.gamma)
            
            
            features = {
                'image' : image,
                'prior_map' : prior_map,
                'patient_id' : int(patient_id),
                'slice_num' : slice_num
            }
            #if np.sum(np.argmax(annotation, axis=-1)) > 0:
            yield (features, annotation)
