![Capture d’écran du 2025-02-08 15-23-18](https://github.com/user-attachments/assets/648926c6-f0ee-4c81-8a38-f929d842ae23)## Summary

Several transformer based models were explored for continuous and discrete emotions prediction using EEV: https://github.com/google-research-datasets/eev and LIRIS: https://liris-accede.ec-lyon.fr/ datasets. 
We used VIT, STAM, TIMESFORMER, VIVIT, audio spectorgam transformers
















Main files:

- updated main file: test_sum.py
- args.py: used arguments
- data_manager.py: implemented functions to process the data
- experiments_manager.py : implemented functions to run experiments
- liris_data_helper.py : functions used to manage/read liris-dataset
- eev_data_helper.py : functions used to manage/read eev-dataset
- save_data: for saving the data in a binary file
- utils.py: include all the used librairies/files
- video_dataset: used to create liris dataset in a Dataset class.
