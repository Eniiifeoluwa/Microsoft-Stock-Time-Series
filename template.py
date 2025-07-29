import os
project_name = 'Microsoft_Stock'
dirs = ['data/raw',
        'data/processed',
        'notebooks',
        'saved_model',
        'src'

]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok= True)
    with open(os.path.join(dir_, '.gitkeep'), 'w') as f:
        pass
file_names = ['params.yaml',
              'dvc.yaml',
              '.gitignore',
              'src/__init__.py'
    
]
for file in file_names:
    with open(file, 'w') as r:
        pass