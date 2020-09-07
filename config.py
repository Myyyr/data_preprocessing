
default_parameters = {
    'dataset' : "tcia" ,#None,
    'data_dir' : None,
    'input_mode' : 'slice',
    
    'model_name' : None,
    'model_dir' : None,

    'lr' : None,
    'final_lr' : None,
    
    'verbose' : 1,
    'train_batch_size' : 10,
    'valid_batch_size' : 10,
    'nb_epochs' : 60,
    'data_augmentation' : False,
    'split_name' : 'split_1',
    'train_splits' : [1,2,3,4],
    'valid_splits' : [0],
    'test_splits' : [],

    'callbacks' : ['save_model', 'csvlogger', 'tensorboard'],

    'problem_type' : 'multiclass',
    'add_ambiguity' : False,
    'final_activation' : 'softmax',
    'annotation_proportion' : 1.0,
    
    'relabel_first': False,
    'relabel_nb_steps': 1,
    'relabel_threshold': 0,
}
