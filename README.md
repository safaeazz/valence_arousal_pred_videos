## Summary: Evoked emotion recognition from videos.

You will find in this repo, several transformer based models were explored for continuous and discrete emotions prediction using EEV: https://github.com/google-research-datasets/eev and LIRIS: https://liris-accede.ec-lyon.fr/ datasets. 
We used VIT, STAM, TIMESFORMER, VIVIT, audio spectorgam transformers


![Capture d’écran du 2025-02-08 15-23-18](https://github.com/user-attachments/assets/648926c6-f0ee-4c81-8a38-f929d842ae23)


You will find a fusion approach in this experiment to predict valence and arousal given a video clip as an input. A spatio-temporal transformer is used for video input and audio transformer encoder for audio input. The fusion was performed at te feature level (learning a shared representation). The table bellow presents the results (using pearson correlation) obtained comparing three fusion strategies on the LIRIS (EIMT’16) DATASET. Namely, additive, multiplicative and late fusion. The best results were obtained using the multiplication technique with a PCC of 0.42 for the arousal prediction and a PCC of 0.37 for the valence target. The fact that additive way is not performing well, could be explained by low correlation between the image and audio modalities and therefore the two modalities are not equivalent or redundant.




![Capture d’écran du 2025-02-08 15-39-24](https://github.com/user-attachments/assets/7f45670e-2c74-459b-bc77-4cb1e65a1707)



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

REF

[7] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri.Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 4489–4497, 2015.

[8] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. arXiv preprint arXiv:1406.2199, 2014

[9] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2625–2634, 2015.
