# Data Augmentation 

Based in the code provided by 
> Xiang Dai and Heike Adel. 2020. An Analysis of Simple Data Augmentation for Named Entity Recognition. In COLING, Online.


## Prepare the dataset
Note that the given dataset in data/ contains only sample files, showing the needed format
~~~
cp /data_path/* data/
~~~

## Prepare the pretrained models
Download from hugging face the desired pretrained model and save it in data


## Experiments
### No augmentation
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/pretrained_path --result_filepath baseline.json
~~~
### Label-wise token replacement
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/pretrained_path --augmentation LwTR --result_filepath lwtr.json
~~~


### Shuffle within segments
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/pretrained_path --augmentation SiS --result_filepath sis.json
~~~
