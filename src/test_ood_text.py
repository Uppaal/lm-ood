import os
import sys
import time
import faiss
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from utils.model_utils import load_model
from utils.dataset_utils import DatasetUtil
from vizualization.plot_utils import plot_historgram
from setup import run_configs, paths, Map, logger
sys.path.extend([os.path.join(paths.root_path, source_path) for source_path in os.listdir(paths.root_path)])# if not (paths.root_path + source_path) in sys.path])

if run_configs.machine == 'local':
    from src.utils.test_utils import get_measures
else:
    from utils.test_utils import get_measures

debug_mode = run_configs.debug_mode
if run_configs.set_seed:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.random.manual_seed(0)
    torch.random.seed()


def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    logger.info(f'ID Dataset: {args.id_dataset}; OOD Dataset: {args.ood_dataset}. \nStarting processing... ')
    train_loader = None

    in_set, out_set = Map(), Map()
    if args.id_dataset == args.ood_dataset:

        data_util = DatasetUtil(dataset_name=args.id_dataset)
        dataset = data_util.get_dataset(args.id_dataset, split=None)
        logger.info(f'Dataset {args.id_dataset} loaded. Starting pre-processing...')
        list_of_id_classes = list(range(len(data_util.id_classes)))

        train_dataset = dataset[data_util.dataset_to_split[args.id_dataset][0]]  # Needed for KNN and Margin
        if args.id_dataset == 'clinc150':
            id_dataset = data_util.merge_dataset_splits([dataset['val'], dataset['test']])
            ood_dataset = data_util.merge_dataset_splits([dataset['oos_train'], dataset['oos_val'], dataset['oos_test']])
        elif args.id_dataset == 'news-category-modified':
            ood_dataset = data_util.merge_dataset_splits(
                [dataset['ood_train_sampled'], dataset['ood_val_sampled'], dataset['ood_test_sampled']])
        else:
            id_dataset = dataset[data_util.dataset_to_split[args.id_dataset][1]]
            ood_dataset = dataset[data_util.dataset_to_split[args.id_dataset][-1]]

        # Get tensors and dataloaders
        in_set.classes = list_of_id_classes
        out_set.classes = list(set(data_util.dataset_to_labels[args.id_dataset]) - set(list_of_id_classes))
        train_dataset = data_util.get_tensors_for_finetuning(train_dataset)
        id_dataset = data_util.get_tensors_for_finetuning(id_dataset)
        ood_dataset = data_util.get_tensors_for_finetuning(ood_dataset)

    elif args.id_dataset in ['imdb', 'sst2', 'yelp_polarity', '20newsgroups']:
        id_data_util = DatasetUtil(dataset_name=args.id_dataset)

        dataset = id_data_util.get_dataset(args.id_dataset, split=None)
        logger.info(f'Dataset {args.id_dataset} loaded. Starting pre-processing...')

        id_dataset = dataset[id_data_util.dataset_to_split[args.id_dataset][1]]
        id_dataset = id_data_util.get_tensors_for_finetuning(id_dataset)
        in_set.classes = id_data_util.dataset_to_labels[args.id_dataset]

        train_dataset = dataset[id_data_util.dataset_to_split[args.id_dataset][0]]
        train_dataset = id_data_util.get_tensors_for_finetuning(train_dataset)

        ood_data_util = DatasetUtil(dataset_name=args.ood_dataset)
        ood_dataset = ood_data_util.get_dataset(args.ood_dataset, split=None)[ood_data_util.dataset_to_split[args.ood_dataset][1]]
        ood_dataset = ood_data_util.get_tensors_for_finetuning(ood_dataset)

    else:
        logger.info('Current dataset not supported.')

    if args.samples_to_keep is not None:
        train_dataset = train_dataset.select(range(args.samples_to_keep))
        id_dataset = id_dataset.select(range(args.samples_to_keep))
        ood_dataset = ood_dataset.select(range(args.samples_to_keep))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch)
    in_loader = torch.utils.data.DataLoader(id_dataset, batch_size=args.batch)
    out_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch)

    logger.info('Data processing complete.')
    return in_set, out_set, in_loader, out_loader, train_loader # out_set not used


def iterate_data_knn(data_loader, model, index, is_train, num_neighbours, distance, load_embeddings,
                    average_kth_nn_distances, is_id_query=True):
    kth_nn_distances = []
    logger.info(f'In iterate_data_knn, with {len(data_loader)} batches.')

    if load_embeddings:
        if is_train:
            # index = pickle.load(open(f'{paths.temp_output_dir}/{args.id_dataset}/index_x.pkl', 'rb'))
            index = pickle.load(open(f'{paths.temp_output_dir}/dapt-news_20newsgroups/index_x.pkl', 'rb'))
            return None, index
        else:
            if is_id_query:
                # z = np.load(f'{paths.temp_output_dir}/{args.id_dataset}/y_id.npz')['arr_0']
                z = np.load(f'{paths.temp_output_dir}/dapt-news_20newsgroups/y_id.npz')['arr_0']
            else:
                # z = np.load(f'{paths.temp_output_dir}/{args.id_dataset}/y_ood.npz')['arr_0']
                z = np.load(f'{paths.temp_output_dir}/dapt-news_20newsgroups/y_ood.npz')['arr_0']
            logger.info(f'num_neighbours={num_neighbours}')
            scores, _ = index.search(z, num_neighbours)

            if average_kth_nn_distances:
                final_score = np.average(scores, axis=1)
            else:
                final_score = scores[:, -1]

            if distance == 'Euclidean':
                kth_nn_distances.extend(-1 * final_score)
            elif distance == 'Cosine':
                # scores are cosine similarity, so higher value means lesser distance
                kth_nn_distances.extend(-1 * (1 - final_score))

        return np.array(kth_nn_distances), index

    else:

        for b, data in enumerate(data_loader):
            if b % 100 == 0:
                logger.info(f'{b} batches processed.')

            if debug_mode and b > 3:
                break

            with torch.no_grad():
                data = {k: v.to(run_configs.device) for k, v in data.items()}
                input_ids, attention_mask = data['input_ids'], data['attention_mask']
                pred = model(input_ids, attention_mask)
                z = pred.hidden_states[-1][:, 0, :].data.cpu().numpy().astype(np.float32)  # [CLS] embeddings only

            # Normalize vectors by their l2 norm; centers data between 1 and -1 I think
            faiss.normalize_L2(z)
            # z = z / np.linalg.norm(z, ord=2, axis=1)[:, np.newaxis] # Both ways are equivalent

            if is_train:
                index.add(z)
            else:
                # logger.info(f'num_neighbours={num_neighbours}')
                scores, _ = index.search(z, num_neighbours)

                if average_kth_nn_distances:
                    final_score = np.average(scores, axis=1)
                else:
                    final_score = scores[:, -1]

                if distance == 'Euclidean':
                    kth_nn_distances.extend(-1 * final_score)
                elif distance == 'Cosine':
                    # scores are cosine similarity, so higher value means lesser distance
                    kth_nn_distances.extend(-1 * (1 - final_score))

        return np.array(kth_nn_distances), index


def iterate_data_margin(train_data_loader, id_data_loader, ood_data_loader, model,
                        num_neighbours, margin_type, hidden_size, aggregation, load_embeddings):
    # Ref: https://github.com/facebookresearch/LASER/blob/7e60ad2d58a2ac4f0bb722d5cbfbe2dbf584c561/source/paraphrase.py#L74

    def get_normalized_embeddings(data_loader, index, batch_limit=3):
        logger.info(f'Starting {len(data_loader)} batches...')

        embeddings = []
        for b, data in enumerate(data_loader):
            if b % 100 == 0:
                logger.info(f'{b} batches processed.')

            if debug_mode and b > batch_limit:
                break

            with torch.no_grad():
                data = {k: v.to(run_configs.device) for k, v in data.items()}
                input_ids, attention_mask = data['input_ids'], data['attention_mask']
                pred = model(input_ids, attention_mask)
                z = pred.hidden_states[-1][:, 0, :].data.cpu().numpy().astype(np.float32)  # [CLS] embeddings only

            # Normalize vectors by their l2 norm; centers data between 1 and -1 I think
            faiss.normalize_L2(z)
            index.add(z)

            embeddings.extend(z)
        return np.array(embeddings), index

    def process_scores(x, y, index_x, index_y):

        # ID train -> x
        # ID test -> y_id
        # OOD all -> y_ood
        # y = [y_id, y_ood]

        cosine_sim, _ = index_x.search(y, x.shape[0]) # (|Y|, |X|)

        sim_x_to_y, _ = index_y.search(x, num_neighbours)  # For each x, find k closest y
        x_to_y = np.sum(sim_x_to_y, axis=1) / (2 * num_neighbours)  # (|X|, )

        sim_y_to_x, _ = index_x.search(y, num_neighbours)  # For each y, find k closest x
        y_to_x = np.sum(sim_y_to_x, axis=1) / (2 * num_neighbours)  # (|Y|, )

        scores = np.zeros((y.shape[0], x.shape[0]))

        logger.info(f'shapes: x-{x.shape} y-{y.shape}\ncosine_sim-{cosine_sim.shape}\n'
                    f'sim_x_to_y-{sim_x_to_y.shape} x_to_y-{x_to_y.shape}\n'
                    f'sim_y_to_x-{sim_y_to_x.shape} y_to_x-{y_to_x.shape}\n'
                    f'scores-{scores.shape}')

        for i in range(y.shape[0]):
            a = cosine_sim[i]        # (|X|, )
            b = x_to_y + y_to_x[i]   # (|X|, )

            if margin_type == 'Absolute':
                scores[i] = a
            elif margin_type == 'Distance':
                scores[i] = a-b
            elif margin_type == 'Ratio':
                scores[i] = a/b

        if aggregation == 'avg':
            aggregated_scores = np.average(scores, axis=1)
        elif aggregation == '1NN':
            aggregated_scores = np.amax(scores, axis=1)  # Nearest neighbour/ 'min'
        elif aggregation == 'max':
            aggregated_scores = np.amin(scores, axis=1)  # Least similar point/ 'max'
        elif aggregation == 'knn':
            # num_neighbours = scores.shape[-1]  #|X|
            # print('HERREEEEE', num_neighbours)
            logger.info(f'num_neighbours={num_neighbours}')
            aggregated_scores = np.sort(scores, axis=1)[:, -num_neighbours]  # np.sort does ascending order

        # Cosine distance = 1 - Cosine Similarity (However this dist breaks triangle inequality).
        # Flip the sign to ensure OOD scores are less than ID
        return -1 * (1 - aggregated_scores)

    def plot_nearest_and_farthest_neighbours_averaged():

        for name, weight in model.named_parameters():  # 12 layers + 1 classifier
            if name == 'module.classifier.dense.weight':
                break
                # activations = pred.hidden_states[-1][:, 0, :]  # (N, 768)
                # unit_contribution = activations @ weight  # (N, 768)
                # We are using the weights from the classifier.dense layer, but the outputs (or activations)
                # are from after the last transformer layer (i.e. just before classifier.dense).
                # I am going with this for now since the visualization is roughly meant as a heuristic, and
                # getting the outputs from within the classifier layer (right after classifier.dense) doesn't
                # seem trivial. TODO: Change this later.

                '''
                pred.hidden_states -> 13 (embedding layer + 12 transformer layers)
                module.roberta.encoder.layer.11.output.dense.weight torch.Size([768, 3072])
                module.roberta.encoder.layer.11.output.dense.bias torch.Size([768])
                module.roberta.encoder.layer.11.output.LayerNorm.weight torch.Size([768])
                module.roberta.encoder.layer.11.output.LayerNorm.bias torch.Size([768])
                module.classifier.dense.weight torch.Size([768, 768])
                module.classifier.dense.bias torch.Size([768])
                module.classifier.out_proj.weight torch.Size([11, 768])
                module.classifier.out_proj.bias torch.Size([11])
                '''

        _, _indices_id = index_x.search(y_id, x.shape[0])
        _, _indices_ood = index_x.search(y_ood, x.shape[0])

        unit_contribution_id_query = y_id @ weight.detach().numpy()
        unit_contribution_ood_query = y_ood @ weight.detach().numpy()
        ordering = np.argsort(np.average(unit_contribution_id_query, axis=0))
        ordering_ood = np.argsort(np.average(unit_contribution_ood_query, axis=0))

        mean_unit_contribution_id_query = np.average(unit_contribution_id_query, axis=0)
        stddev_unit_contribution_id_query = np.std(unit_contribution_id_query, axis=0)
        mean_unit_contribution_id_nearest_neighbour = np.average(x[_indices_id[:][0]] @ weight.detach().numpy(), axis=0)
        mean_unit_contribution_id_farthest_neighbour = np.average(x[_indices_id[:][-1]] @ weight.detach().numpy(), axis=0)

        mean_unit_contribution_ood_query = np.average(unit_contribution_ood_query, axis=0)
        stddev_unit_contribution_ood_query = np.std(unit_contribution_ood_query, axis=0)
        mean_unit_contribution_ood_nearest_neighbour = np.average(x[_indices_ood[:][0]] @ weight.detach().numpy(), axis=0)
        mean_unit_contribution_ood_farthest_neighbour = np.average(x[_indices_ood[:][-1]] @ weight.detach().numpy(), axis=0)

        # Plot 1 - ID query mean and variance
        plt.plot(mean_unit_contribution_id_query[ordering], label='ID Query Point', alpha=0.5)
        plt.fill_between(list(range(768)), mean_unit_contribution_id_query[ordering] + stddev_unit_contribution_id_query[ordering],
                         mean_unit_contribution_id_query[ordering] - stddev_unit_contribution_id_query[ordering], facecolor='C0', alpha=0.1)
        plt.plot(mean_unit_contribution_id_nearest_neighbour[ordering], label='Nearest ID Neighbour', alpha=0.3)
        plt.plot(mean_unit_contribution_id_farthest_neighbour[ordering], label='Farthest ID Neighbour', alpha=0.3)
        # plt.ylim([-0.03, 0.03])
        plt.xlabel('Units from Penultimate Layer ([CLS]) (Sorted by mean activation of ID query point)')
        plt.ylabel('L2 normalized outputs * Weights')
        title_string = f'20 NewsGroups - Averaged unit contributions for ID query point'
        plt.title(title_string)
        plt.legend()
        plt.grid()
        plt.savefig(f'{paths.output_plot_dir}/{title_string}.png')
        plt.show()

        # Plot 2 - OOD query mean and variance (sorted by ID ordering)
        plt.plot(mean_unit_contribution_ood_query[ordering], label='OOD Query Point', alpha=0.5)
        plt.fill_between(list(range(768)), mean_unit_contribution_ood_query[ordering] + stddev_unit_contribution_ood_query[ordering],
                         mean_unit_contribution_ood_query[ordering] - stddev_unit_contribution_ood_query[ordering], facecolor='C0', alpha=0.1)
        plt.plot(mean_unit_contribution_ood_nearest_neighbour[ordering], label='Nearest Neighbour', alpha=0.5)
        plt.plot(mean_unit_contribution_ood_farthest_neighbour[ordering], label='Farthest Neighbour', alpha=0.2)
        # plt.ylim([-0.03, 0.03])
        plt.xlabel('Units from Penultimate Layer ([CLS]) (Sorted by mean activation of ID query point)')
        plt.ylabel('L2 normalized outputs * Weights')
        title_string = f'20 NewsGroups - Averaged unit contributions for OOD query point'
        plt.title(title_string)
        plt.legend()
        plt.grid()
        plt.savefig(f'{paths.output_plot_dir}/{title_string}_ordered_by_id.png')
        plt.show()

        # Plot 3 - OOD query mean and variance (sorted by OOD ordering)
        plt.plot(mean_unit_contribution_ood_query[ordering_ood], label='OOD Query Point', alpha=0.5)
        plt.fill_between(list(range(768)), mean_unit_contribution_ood_query[ordering_ood] + stddev_unit_contribution_ood_query[ordering_ood],
                         mean_unit_contribution_ood_query[ordering_ood] - stddev_unit_contribution_ood_query[ordering_ood], facecolor='C0', alpha=0.1)
        plt.plot(mean_unit_contribution_ood_nearest_neighbour[ordering_ood], label='Nearest Neighbour', alpha=0.7)
        plt.plot(mean_unit_contribution_ood_farthest_neighbour[ordering_ood], label='Farthest Neighbour', alpha=0.3)
        plt.ylim([-0.03, 0.03])
        plt.xlabel('Units from Penultimate Layer ([CLS]) (Sorted by mean activation of OOD query point)')
        plt.ylabel('L2 normalized outputs * Weights')
        title_string = f'20 NewsGroups - Averaged unit contributions for OOD query point'
        plt.title(title_string)
        plt.legend()
        plt.grid()
        plt.savefig(f'{paths.output_plot_dir}/{title_string}_ordered_by_ood.png')
        plt.show()


    if load_embeddings:
        x = np.load(f'{paths.temp_output_dir}/dapt-news_20newsgroups/x.npz')['arr_0']
        index_x = pickle.load(open(f'{paths.temp_output_dir}/dapt-news_20newsgroups/index_x.pkl', 'rb'))
        y_id = np.load(f'{paths.temp_output_dir}/dapt-news_20newsgroups/y_id.npz')['arr_0']
        y_ood = np.load(f'{paths.temp_output_dir}/dapt-news_20newsgroups/y_ood.npz')['arr_0']
        index_y = pickle.load(open(f'{paths.temp_output_dir}/dapt-news_20newsgroups/index_y.pkl', 'rb'))

        # x = np.load(f'{paths.temp_output_dir}/{args.id_dataset}/x.npz')['arr_0']
        # index_x = pickle.load(open(f'{paths.temp_output_dir}/{args.id_dataset}/index_x.pkl', 'rb'))
        # y_id = np.load(f'{paths.temp_output_dir}/{args.id_dataset}/y_id.npz')['arr_0']
        # y_ood = np.load(f'{paths.temp_output_dir}/{args.id_dataset}/y_ood.npz')['arr_0']
        # index_y = pickle.load(open(f'{paths.temp_output_dir}/{args.id_dataset}/index_y.pkl', 'rb'))
        logger.info('Loaded embeddings.')
    else:
        index_x = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        index_y = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)

        logger.info(f'Calculating indices for train data...')
        x, index_x = get_normalized_embeddings(train_data_loader, index_x, batch_limit=40)
        np.savez_compressed(f'{paths.temp_output_dir}/x.npz', x)
        pickle.dump(index_x, open(f'{paths.temp_output_dir}/index_x.pkl', 'wb'))

        logger.info(f'Calculating indices for ID data...')
        y_id, index_y = get_normalized_embeddings(id_data_loader, index_y, batch_limit=30)
        np.savez_compressed(f'{paths.temp_output_dir}/y_id.npz', y_id)

        logger.info(f'Calculating indices for OOD data...')
        y_ood, index_y = get_normalized_embeddings(ood_data_loader, index_y, batch_limit=50)
        np.savez_compressed(f'{paths.temp_output_dir}/y_ood.npz', y_ood)
        pickle.dump(index_y, open(f'{paths.temp_output_dir}/index_y.pkl', 'wb'))

    # plot_nearest_and_farthest_neighbours_averaged()
    # logger.info(f'Plotted nearest and farthest point activations.')

    logger.info(f'Calculating margin scores ({margin_type} Margin and {aggregation} aggregation)...')
    scores_total = process_scores(x, np.concatenate([y_id, y_ood]), index_x, index_y)
    in_scores = scores_total[:len(y_id)]
    out_scores = scores_total[len(y_id):]
    assert len(out_scores) == len(y_ood)

    return in_scores, out_scores

def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()

    for b, data in enumerate(data_loader):
        if b % 100 == 0:
            logger.info(f'{b} batches processed.')

        if debug_mode and b > 3:
            break

        with torch.no_grad():
            data = {k: v.to(run_configs.device) for k, v in data.items()}
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            pred = model(input_ids, attention_mask)
            logits = pred.logits

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []

    for b, data in enumerate(data_loader):
        if b % 100 == 0:
            logger.info(f'{b} batches processed.')

        if debug_mode and b > 3:
            break

        with torch.no_grad():
            data = {k: v.to(run_configs.device) for k, v in data.items()}
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            pred = model(input_ids, attention_mask)
            logits = pred.logits

            conf = temper * torch.logsumexp(logits / temper, dim=1)  # (N, )
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_ppl(data_loader, model, stride=1):

    max_length = model.config.max_position_embeddings
    seq_len = data_loader.dataset['input_ids'].size(1)
    input_ids_full = data_loader.dataset['input_ids']
    lls = []
    #
    # for b, data in enumerate(data_loader):
    #     if b % 100 == 0:
    #         logger.info(f'{b} batches processed.')
    #
    #     if debug_mode and b > 3:
    #         break
    #
    #     with torch.no_grad():
    #         data = {k: v.to(run_configs.device) for k, v in data.items()}
    #         input_ids, attention_mask = data['input_ids'], data['attention_mask']

    for i in range(1, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = input_ids_full[:, begin_loc:end_loc].to(run_configs.device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100  # Masks tokens already calculated in previous passes

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * stride  # -log likelihood = cross entropy

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i)
    # return ppl.item(), torch.stack(lls).detach().cpu().numpy()
    return np.array(ppl)


def run_eval(model, in_loader, out_loader, logger, args, train_loader):
    # switch to evaluate mode
    model.eval()

    logger.info(f"Running test for {args.score}...")
    logger.info(f'({len(in_loader)} ID batches and {len(out_loader)} OOD batches)')

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)

    elif args.score == 'PPL':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_ppl(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_ppl(out_loader, model)

    elif args.score == 'KNN':
        if args.knn_distance == 'Euclidean':
            index = faiss.IndexFlatL2(model.config.hidden_size)  # 768
        elif args.knn_distance == 'Cosine':
            # Ref: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
            # Ref: https://github.com/facebookresearch/faiss/issues/95
            index = faiss.index_factory(model.config.hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)

        # k = min(args.num_neighbours, args.batch)
        k = args.num_neighbours

        _, index = iterate_data_knn(train_loader, model, index, is_train=True, num_neighbours=k, distance=args.knn_distance,
                                    load_embeddings=args.load_embeddings, average_kth_nn_distances=args.average_kth_nn_distances)

        logger.info("Processing in-distribution data...")
        in_scores, index = iterate_data_knn(in_loader, model, index, is_train=False, num_neighbours=k, distance=args.knn_distance,
                                            load_embeddings=args.load_embeddings, is_id_query=True,
                                            average_kth_nn_distances=args.average_kth_nn_distances)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_knn(out_loader, model, index, is_train=False, num_neighbours=k, distance=args.knn_distance,
                                         load_embeddings=args.load_embeddings, is_id_query=False,
                                         average_kth_nn_distances=args.average_kth_nn_distances)

    elif args.score == 'Margin':

        # cos(A, B) = \frac{A.B}{||A|| ||B||} = \frac{\Sigma A_i B_i}{\sqrt{\Sigma A_i^2} \sqrt{\Sigma B_i^2}}
        # In simpler terms, cosine similarity = inner product (A, B) / L2_norm(A) L2_norm(B)
        # So using FAISS, normalize all vectors by their L2 norm, then find Inner Product.
        # index = faiss.index_factory(model_config.hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)

        k = min(args.num_neighbours, args.batch)
        # k = args.num_neighbours

        logger.info("Processing data...")
        in_scores, out_scores = iterate_data_margin(train_data_loader=train_loader, id_data_loader=in_loader, ood_data_loader=out_loader,
                                                    model=model, hidden_size=model.config.hidden_size, load_embeddings=args.load_embeddings,
                                                    num_neighbours=k, margin_type=args.margin_type, aggregation=args.aggregation)

    elif args.score == 'Energy': # Needs tuning of temperature
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)

    else:
        raise ValueError("Unknown score type {}".format(args.score))

    # Plot histogram of ID vs OOD scores (i.e. frequency of points at each score value)
    np.save(f'{paths.temp_output_dir}/in_scores_{args.score}_{args.id_dataset}_to_{args.ood_dataset}.npy', in_scores)
    np.save(f'{paths.temp_output_dir}/out_scores_{args.score}_{args.id_dataset}_to_{args.ood_dataset}.npy', out_scores)
    # confs_id = torch.logsumexp(torch.randn(N, C_id), 1).data.cpu().numpy()

    plot_historgram(scores=[in_scores, out_scores], labels=['ID data', 'OOD data'],
                    x_label='Score', y_label='Frequency',
                    title=f'{args.id_dataset} to {args.ood_dataset} - {args.score} Score',
                    # title=f'{args.margin_type} Margin ({args.aggregation} Aggregation, {args.num_neighbours} Neighbours)',
                    savename=f'../plots/{args.id_dataset} to {args.ood_dataset} - {args.score} Score ({args.model_name}).png')
                    # savepath=f'{paths.output_plot_dir}/{args.margin_type} Margin ({args.aggregation} Aggregation, {args.num_neighbours} Neighbours).png')

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    logger.info(f'In Scores: min={np.min(in_scores)} max={np.max(in_scores)} average={np.average(in_scores)}')
    logger.info(f'Out Scores: min={np.min(out_scores)} max={np.max(out_scores)} average={np.average(out_scores)}')
    logger.info(f'In score samples: {in_scores[:5]}')
    logger.info(f'Out score samples: {out_scores[:5]}')

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    # results = cal_metric(known=np.squeeze(in_examples), novel=np.squeeze(out_examples))
    # logger.info(results)

    logger.info(f'============Results for {args.score} (ID data: {args.id_dataset} OOD data: {args.ood_dataset} Model: {args.model_name})============')
    logger.info(f'AUROC: {auroc}')
    logger.info(f'AUPR (In): {aupr_in}')
    logger.info(f'AUPR (Out): {aupr_out}')
    logger.info(f'FPR95: {fpr95}')

    auroc, aupr_in, aupr_out, fpr95 = get_measures(out_examples, in_examples)
    logger.info(f'============Reversed Results for {args.score} (ID data: {args.id_dataset} OOD data: {args.ood_dataset} Model: {args.model_name})============')
    logger.info(f'AUROC: {auroc}')
    logger.info(f'AUPR (In): {aupr_in}')
    logger.info(f'AUPR (Out): {aupr_out}')
    logger.info(f'FPR95: {fpr95}')

    logger.info(f'Running with args {vars(args)}')


def main(args):
    torch.backends.cudnn.benchmark = True
    if debug_mode:
        args.batch = 8

    in_set, _, in_loader, out_loader, train_loader = make_id_ood(args, logger)

    logger.info(f"Loading model: {args.model_name} from {paths.MODEL_DIR}.")
    model, _ = load_model(model_name=f'finetuned_models/{args.model_name}', num_labels=len(in_set.classes))

    model = model.to(run_configs.device)

    start_time = time.time()
    run_eval(model=model, in_loader=in_loader, out_loader=out_loader, train_loader=train_loader, logger=logger, args=args)
    end_time = time.time()
    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    args = Map()
    args.score = 'MSP'             # 'Energy' 'MSP' 'KNN' 'Margin'
    args.id_dataset = 'sst2'   # 'news-category-modified' 'clinc150' '20newsgroups-open-set'
    args.ood_dataset = '20newsgroups'  # 'news-category-modified' 'clinc150' '20newsgroups-open-set'

    if run_configs.debug_mode:
        args.model_name = f'roberta_base'
    else:
        # args.model_name = f'roberta_base_{args.id_dataset}'
        # args.model_name = f'roberta_base_tapt-{args.id_dataset}-finetuned'
        args.model_name = f'roberta_base_{args.id_dataset}_contrastive_margin'

    args.batch = 1 if run_configs.debug_mode else 256
    args.samples_to_keep = 5 if run_configs.debug_mode else None
    args.temperature_energy = 1
    args.name = 'temp'
    args.logdir = f'{paths.temp_output_dir}/test_log'
    args.workers = 8

    args.load_embeddings = False
    args.num_neighbours = 1       # For k-NN (Yiyou's paper shows between 10 and 100 is optimal) # Margin paper uses k=4
    args.knn_distance = 'Cosine'  # 'Cosine' 'Euclidean'
    args.average_kth_nn_distances = False  # If true, averages scores of k nearest neighbours
    args.margin_type = 'Absolute' # 'Absolute' 'Distance' 'Ratio'
    args.aggregation = '1NN'      # 'avg' 'max' '1NN' 'knn'

    logger.info(f'Running with args {vars(args)}')
    main(args)
