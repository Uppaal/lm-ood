import sys
import umap
import umap.plot
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/afs/cs.wisc.edu/u/u/p/uppaal/Code/ood-in-text/src')
from setup import paths, logger, run_configs

if run_configs.machine == 'local':
    from src.utils.model_utils import load_model
    from src.utils.dataset_utils import DatasetUtil
else:
    from utils.model_utils import load_model
    from utils.dataset_utils import DatasetUtil


def save_final_layer_embeddings(processed_dataset, data_util, model_name, is_id,
                                debug_mode=False, keep_fraction_of_data=False,
                                get_cls_embedding=True, penultimate_layer='dense_input',
                                get_sentence_embedding_from_library=False,
                                batch_size=64):
    """

    :param processed_dataset:
    :param data_util:
    :param model_name:
    :param is_id:
    :param debug_mode:
    :param keep_fraction_of_data:
    :param get_cls_embedding:
    :param penultimate_layer: 'dense_input' is the output just before the classifier head. 'dense_output' is after the first dense layer in the classifier head.
    :param get_sentence_embedding_from_library:
    :return:
    """

    labels = processed_dataset['label']
    dataset_name = data_util.dataset_name

    if debug_mode:
        batch_size = 1

    if get_sentence_embedding_from_library:
        # Results don't seem to match using [CLS]. Leave this one for now.

        # This library uses Sentence-BERT/RoBERTa, a modification of the pretrained BERT network
        # that use siamese and triplet network structures to derive semantically meaningful
        # sentence embeddings that can be compared using cosine-similarity.
        # This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa
        # to about 5 seconds with SBERT, while maintaining the accuracy from BERT.

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('roberta_base')

        sentence1_key, sentence2_key = data_util.dataset_to_keys[data_util.dataset_name][:2]
        if sentence2_key is None:
            sentences = [x[sentence1_key] for x in processed_dataset]
        else:
            sentences = [x[sentence1_key] + ' ' + x[sentence2_key] for x in processed_dataset]

        sentence_embeddings = model.encode(sentences)
        final_hidden_state = sentence_embeddings



    else:
        if penultimate_layer == 'dense_input':
            # It doesn't matter if we use LMHead or SeqClassification head. In either case,
            # penultimate layer embeddings are used so viz are consistent (empircially tested as well).
            model, _ = load_model(model_name=model_name)
            model = torch.nn.DataParallel(model)
        else:
            from contra_ood.model import RobertaForSequenceClassification
            model_path = paths.MODEL_DIR + f'/{model_name}'
            model = RobertaForSequenceClassification.from_pretrained(model_path)

        model = model.to(run_configs.device)
        dataloader = data_util.get_dataloader(processed_dataset, batch_size=batch_size)
        logger.info(f'{len(dataloader)} batches ({len(processed_dataset)} points with batch size {batch_size})')

        del processed_dataset

        final_hidden_state = []
        for i, batch in enumerate(dataloader):
            model.zero_grad()
            if debug_mode and i > 5:
                break

            if i % 50 == 0:
                logger.info(f'Batch {i}')
            with torch.no_grad():
                batch = {k: v.to(run_configs.device) for k, v in batch.items()}

                if penultimate_layer == 'dense_input':
                    pred = model(batch['input_ids'], batch['attention_mask'])

                    # Last 768 dim layer, before C dimemsional logits
                    if get_cls_embedding:
                        final_hidden_state.append(pred.hidden_states[-1][:, 0, :].data.cpu().numpy())
                    else:
                        final_hidden_state.append(pred.hidden_states[-1].data.cpu().numpy())

                else:
                    outputs = model.roberta(batch['input_ids'], batch['attention_mask'])
                    sequence_output = outputs[0]
                    logits, pooled = model.classifier(sequence_output)
                    final_hidden_state.append(pooled.data.cpu().numpy())

            torch.cuda.empty_cache()

        final_hidden_state = np.concatenate(final_hidden_state)

        if not get_cls_embedding:
            N, L, D = final_hidden_state.shape
            final_hidden_state = final_hidden_state.reshape((N, L * D))
        else:
            N, D = final_hidden_state.shape

        if debug_mode:
            labels = labels[:N]

    # Keep 10% of original data to plot, maintaining class-wise distribution (This is to avoid memory issues)
    if not debug_mode and keep_fraction_of_data:
        indices_to_keep = []
        for class_idx in range(max(data_util.dataset_to_labels[dataset_name])):
            indicies_for_class = np.where(labels==class_idx)[0]
            indices_to_keep.append(np.random.choice(indicies_for_class, size=int(len(indicies_for_class) * 0.1), replace=False))
        indices_to_keep = np.concatenate(indices_to_keep)
        labels = labels[indices_to_keep]
        final_hidden_state = final_hidden_state[indices_to_keep]
        # For NewsCategory: # From 200853 points, plotted 19963

    if is_id:
        np.savez_compressed(f'{paths.temp_output_dir}/final_hidden_state_{dataset_name}_id.npz', final_hidden_state)
        np.savez_compressed(f'{paths.temp_output_dir}/labels_{dataset_name}_id.npz', labels)
    else:
        np.savez_compressed(f'{paths.temp_output_dir}/final_hidden_state_{dataset_name}_ood.npz', final_hidden_state)
        np.savez_compressed(f'{paths.temp_output_dir}/labels_{dataset_name}_ood.npz', labels)
    logger.info('Saved embeddings.')


def _umap_visualization(x, y, filename, metric='cosine', min_dist=0.1, n_neighbors=10, densmap=True,
                       id_ood_plot=False):
    '''
    Applies a UMap Visualization on the tensors of the given dataset. Similar to t-SNE, but better and faster.
    Ref: https://github.com/lmcinnes/umap
    :param x: Typically dataset['input_ids']/ layer embeddings (N, L, D). Must be numpy array.
    :param y: Typically dataset['label']. Must be numpy array.
    :param filename: Filename to save plots with.
    :param metric: 'correlation' or 'cosine' work better. Default is 'euclidean'
    :param min_dist:
    :param n_neighbors:
    :param densmap:
    :return:
    '''

    logger.info('Starting UMap visualization...')
    embedding = umap.UMAP(densmap=densmap, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(x)
    logger.info('Embeddings calculated.')

    if id_ood_plot:
        mask = np.zeros_like(y)
        if 'news-category-hf' in filename:
            mask[y > 4] = 1
        elif 'news-category-modified' in filename:
            mask[y > 16] = 1
        elif 'dbpedia' in filename:
            mask[y > 3] = 1
        elif 'clinc150' in filename:
            mask[y == 150] = 1
        elif '20newsgroups-open-set' in filename:
            mask[y > 10] = 1

        umap.plot.points(embedding, labels=mask, color_key={0: 'red', 1: 'yellow'})
        plt.savefig(f'{paths.output_plot_dir}/{"densmap" if densmap else "umap"}_{filename}_id-vs-ood.png')

    umap.plot.points(embedding, labels=y)
    savename = f'{paths.output_plot_dir}/{"densmap" if densmap else "umap"}_{filename}.png'  # _params-{embedding.metric}-densemap{densmap}-{n_neighbors}neighbours-{min_dist}mindist.png'
    plt.savefig(savename)
    plt.show()
    logger.info(f'Saved plot to {savename}')


def umap_visualization(x_dict, filename, metric='cosine', min_dist=0.1, n_neighbors=10, densmap=True):
    '''
    Applies a UMap Visualization on the tensors of the given dataset. Similar to t-SNE, but better and faster.
    Ref: https://github.com/lmcinnes/umap
    :param x: Typically dataset['input_ids']/ layer embeddings (N, L, D). Must be numpy array.
    :param y: Typically dataset['label']. Must be numpy array.
    :param filename: Filename to save plots with.
    :param metric: 'correlation' or 'cosine' work better. Default is 'euclidean'
    :param min_dist:
    :param n_neighbors:
    :param densmap:
    :return:
    '''

    logger.info('Starting UMap visualization...')
    id_dataset_name = list(x_dict.keys())[0]

    x_test, y_test = [], []
    for k, v in x_dict.items():
        if k != id_dataset_name:
            x_test.append(v)
            y_test.extend([k] * len(v))

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.array(y_test)
    logger.info(f'x: {x_test.shape} y: {y_test.shape}')

    embedding = umap.UMAP(densmap=densmap, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(x_dict[id_dataset_name])
    umap.plot.points(embedding, labels=np.array([id_dataset_name] * len(x_dict[id_dataset_name])))
    savename = f'{paths.output_plot_dir}/{"densmap" if densmap else "umap"}_{filename}_train-only.png'
    plt.title('Training Data Only')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    logger.info(f'Saved plot to {savename}')

    labels = list(set(y_test))
    colours = ['blue', 'green', 'red', 'yellow']
    colour_key = {labels[i]:colours[i] for i in range(len(labels))}

    embedding = umap.UMAP(densmap=densmap, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(x_test)
    umap.plot.points(embedding, labels=y_test, color_key=colour_key)
    savename = f'{paths.output_plot_dir}/{"densmap" if densmap else "umap"}_{filename}_test-only.png'
    plt.title('ID and OOD Test Data Only')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    logger.info(f'Saved plot to {savename}')



def get_umap_plots_from_embeddings(dataset_names, filename=None):
    final_hidden_state = {}
    id_dataset_name = dataset_names[0]
    if filename is None:
        filename = id_dataset_name

    final_hidden_state[f'train_{id_dataset_name}'] = np.load(f'{paths.temp_output_dir}/final_hidden_state_{id_dataset_name}_id.npz')['arr_0']

    for ood_dataset_name in dataset_names[1:]:
        key = f'id_{ood_dataset_name}' if ood_dataset_name == id_dataset_name else f'ood_{ood_dataset_name}'
        final_hidden_state[key] = np.load(f'{paths.temp_output_dir}/final_hidden_state_{ood_dataset_name}_ood.npz')['arr_0']

    umap_visualization(x_dict=final_hidden_state, filename=filename)
    logger.info('Done.')


def process_dataset(dataset_name, is_id, split_idx=None):
    data_util = DatasetUtil(dataset_name=dataset_name, max_length=run_configs.max_seq_len)
    dataset = data_util.get_dataset(dataset_name, split=None)

    if split_idx is None:
        split_idx = 0 if is_id else 1

    dataset_split = data_util.get_tensors_for_finetuning(dataset[data_util.dataset_to_split[dataset_name][split_idx]])
    return dataset_split, data_util


if __name__ == "__main__":
    id_dataset = '20newsgroups'  # '20newsgroups' 'sst2' 'news-category-modified' 'clinc150'
    ood_datasets = ['rte']
    # ood_datasets = ['sst2', '20newsgroups', 'rte', 'mnli', 'imdb', 'multi30k', 'news-category-modified', 'clinc150']

    penultimate_layer = 'dense_output'  # 'dense_input' 'dense_output'

    for epoch in range(10):
        # model_name = f'finetuned_models/ce/id_20ng/roberta_base_20newsgroups_ce_epoch-{epoch}'
        model_name = 'pretrained_models/roberta_base'

        id_dataset_tensors, id_data_util = process_dataset(id_dataset, is_id=True)
        save_final_layer_embeddings(processed_dataset=id_dataset_tensors, data_util=id_data_util, is_id=True,
                                    get_cls_embedding=True, model_name=model_name,
                                    penultimate_layer=penultimate_layer, debug_mode=run_configs.debug_mode)

        for ood_dataset in ood_datasets:
            ood_dataset_tensors, ood_data_util = process_dataset(ood_dataset, is_id=False)
            save_final_layer_embeddings(processed_dataset=ood_dataset_tensors, data_util=ood_data_util, is_id=False,
                                        get_cls_embedding=True, model_name=model_name, debug_mode=run_configs.debug_mode,
                                        penultimate_layer=penultimate_layer)

        get_umap_plots_from_embeddings([id_dataset] + ood_datasets, filename=f'20ng_ce_epoch-{epoch}')
