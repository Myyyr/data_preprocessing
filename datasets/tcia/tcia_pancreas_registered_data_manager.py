
            


class PancreasRegisteredPriorDataManager(DataManager):

    def __init__(self, data_dir, random_seed, prior_map, validation_data_proportion=0.2, **_ignored):
        super().__init__(data_dir, random_seed, validation_data_proportion)

        self.nb_classes = 2
        self.input_size = (512, 512, 1)
        #self.mean_value = 48.95255
        self.mean_value = 18.6461
        #self.std_value = 6.6069
        self.std_value = 21.5892

        
        self.info_filename = "Info_pancreas_position.yaml"
        self._prior_map = np.load(prior_map)
        
        self.output_types = ({
            'image' : tf.float32,
            'prior_map' : tf.float32,
            'patient_id' : tf.int32,
            'slice_num' : tf.int32
        }, tf.uint8)

        self.output_shapes = ({
            'image' : tf.TensorShape([None, None, 1]),
            'prior_map' : tf.TensorShape([None, None, self.nb_classes]),
            'patient_id' : tf.TensorShape([]),
            'slice_num' : tf.TensorShape([]),
        }, tf.TensorShape([None, None, self.nb_classes]))
        
        
    def gen(self, examples_names):
        for i, filename in enumerate(examples_names):
            patient_id, slice_num = filename.split('.')[0].split('/')
            slice_num = int(slice_num)
            image = np.load(os.path.join(self.image_path, filename))[:,:,1]
            
            bones_seg = cv2.Canny(image[2:-2,2:-2],100,150)
            if np.sum(bones_seg) > 0:
                BB = get_2D_bbox(bones_seg, margin=0)
            else:
                BB = ((0,self.input_size[0]),(0,self.input_size[1]))

            w = BB[1][1]-BB[1][0]
            h = BB[0][1]-BB[0][0]

            top_left_x = BB[1][0]
            top_left_y = BB[0][0]

            
            prior_map = np.stack([np.ones((self.input_size[0], self.input_size[1])),
                                 np.zeros((self.input_size[0], self.input_size[1]))],
                                 axis=-1)

            prior_map[top_left_y:top_left_y+h, top_left_x:top_left_x+w, :] = cv2.resize(self._prior_map, (w,h), interpolation=cv2.INTER_LINEAR)
            
            image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32)
            image = image - self.mean_value
            
            annotation = np.load(os.path.join(self.annotation_path, filename))[:,:,1]

            annotation = tf.keras.utils.to_categorical(annotation, num_classes=2)

            
            features = {
                'image' : image,
                'prior_map' : prior_map,
                'patient_id' : int(patient_id),
                'slice_num' : slice_num
            }

            # if no_zeros:
            #     if np.sum(annotation[:,:,1]) > 0:
            #         yield (features, annotation)
            # else:
            yield (features, annotation)
