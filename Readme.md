## Exploring OOD Detection in Text

Before being able to use this codebase, create the following empty directories: `data`, `models` and `plots`, `temp_outputs`. Add your pretrained models in `models`. 

- To run an implemented OOD detection method (MSP, Energy, kNN),
    - Run `python src/test_ood_text.py`
    - Choosing the specifc scoring method, and setting up the hyperparams is done in the `main` method of the file.
- To train a model with a contrastive objective, and get detection scores, 
    - Run `python src/contra_ood/run.py`
    - Set parameters in the main method of the file.  
- To get a visualization of a specifc dataset (using UMap), run `python src/vizualization/umap_viz.py`
- To finetune or evaluate RoBERTa on a supported dataset, run `python src/training/finetune_roberta.py`