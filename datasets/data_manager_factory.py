from datasets.data_manager_2d import DataManager2D

class DataManagerFactory():

    def __init__(self):
        self.possible_values = ['TCIA_pancreas',
                                'LiTS',
                                'VP']
    
    def get_data_manager(self, name, params):
        if name not in self.possible_values:
            raise ValueError('Name "{}" for DataManager should be one of {}'.format(name, self.possible_values))
        else:                
            if name == 'TCIA_pancreas':
                print('LOADING TCIA Pancreas DATASET')
                return DataManager2D
            if name == 'LiTS':
                print('LOADING LiTS DATASET')
                return DataManager2D
            if name == 'VP':
                print('LOADING VP DATASET')
                return DataManager2D
