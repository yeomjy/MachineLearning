import glob
import numpy as np
from pathlib import Path


# path: test_dir, train_dir, valid_dir
def make_list(path: Path):
    np.random.seed(0)

    kidney_normal = path / 'Normal' / 'generated' / 'Kidney'
    kidney_abnormal = path / 'Wilms_Tumor' / 'generated' / 'Kidney'


    k_normal = list(kidney_normal.glob('*.png'))
    k_abnormal = list(kidney_abnormal.glob('*.png'))

    indexes = np.arange(0, len(k_normal) + len(k_abnormal))
    np.random.shuffle(indexes)
    n = len(k_abnormal)
    list_kidney = k_abnormal + k_normal

    with open(path.parent / (path.name + '_kidney.txt'), 'w') as f:
        for i in indexes:
            if i < n:
                f.write(f'{list_kidney[i]} abnormal\n')
            else:
                f.write(f'{list_kidney[i]} normal\n')




if __name__ == '__main__':

    base_dir = Path('data')
    test_dir = base_dir / 'Test'
    train_dir = base_dir / 'Train'
    valid_dir = base_dir / 'Valid'

    make_list(test_dir)
    make_list(train_dir)
    make_list(valid_dir)

    # Liver classification Model: Liver_cancer, Normal 
    # Kidney classification Model: Wilms_Tumor, Normal


    
    

    

