import os
import sys
import time
import faiss
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegressionCV
from transformers import RobertaConfig, RobertaForSequenceClassification

sys.path.append(os.getcwd() + '/..')
from src.setup import machine, device, MODEL_DIR, root_path, Map, logger, set_seed, debug_mode, run_configs
from old import log

if run_configs.machine == 'local':
    from src.utils.dataset_utils import DatasetUtil
    from src.utils.test_utils import get_measures
    from src.utils.mahalanobis_lib import get_Mahalanobis_score
else:
    from utils.dataset_utils import DatasetUtil
    from utils.test_utils import get_measures
    from utils.mahalanobis_lib import get_Mahalanobis_score

torch.cuda.empty_cache() # Save memory

if set_seed:
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

        if args.id_dataset == 'dbpedia-local':
            list_of_id_classes = [0, 1, 2, 3]
        elif args.id_dataset in ['news-category-hf', 'news-category-modified']:
            # list_of_id_classes = [0, 1, 2, 3, 4]
            list_of_id_classes = [data_util.class_mappings_original_to_modified[x] for x in data_util.id_classes]

        # Seperate into ID and OOD
        # id_dataset = data_util.get_data_from_classes(dataset=dataset, list_of_classes=list_of_id_classes)
        # ood_dataset = data_util.get_data_from_classes(dataset=dataset, list_of_classes=list(set(data_util.dataset_to_labels[args.id_dataset]) - set(list_of_id_classes)))
        train_dataset = dataset['id_train_sampled']  # Needed for KNN
        id_dataset = dataset['id_test_sampled']
        ood_dataset = data_util.merge_dataset_splits([dataset['ood_train_sampled'], dataset['ood_val_sampled'], dataset['ood_test_sampled']])

        # Get tensors and dataloaders
        in_set.classes = list_of_id_classes
        out_set.classes = list(set(data_util.dataset_to_labels[args.id_dataset]) - set(list_of_id_classes))
        in_loader = data_util.get_dataloader(data_util.get_tensors_for_finetuning(id_dataset), batch_size=args.batch)
        out_loader = data_util.get_dataloader(data_util.get_tensors_for_finetuning(ood_dataset), batch_size=args.batch)
        train_loader = data_util.get_dataloader(data_util.get_tensors_for_finetuning(train_dataset), batch_size=args.batch)

    elif args.id_dataset in ['imdb', 'sst2', 'yelp_polarity']:
        id_data_util = DatasetUtil(dataset_name=args.id_dataset)
        in_loader = id_data_util.get_dataloader(
            id_data_util.get_tensors_end_to_end(split=id_data_util.dataset_to_split[args.id_dataset][-1]), batch_size=args.batch)
        in_set.classes = id_data_util.dataset_to_labels[args.id_dataset]

        ood_data_util = DatasetUtil(dataset_name=args.ood_dataset)
        out_loader = ood_data_util.get_dataloader(
            ood_data_util.get_tensors_end_to_end(split=ood_data_util.dataset_to_split[args.ood_dataset][-1]), batch_size=args.batch)
        out_set.classes = ood_data_util.dataset_to_labels[args.ood_dataset]

    else:
        logger.info('Current dataset not supported.')

    logger.info('data processing complete.')
    return in_set, out_set, in_loader, out_loader, train_loader



def iterate_data_knn(data_loader, model, index, is_train, num_neighbours, distance):
    kth_nn_distances = []
    logger.info(f'In iterate_data_knn, with {len(data_loader)} batches.')

    for b, data in enumerate(data_loader):
        if b % 100 == 0:
            logger.info(f'{b} batches processed.')

        if debug_mode and b > 3:
            break

        with torch.no_grad():
            data = {k: v.to(device) for k, v in data.items()}
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            pred = model(input_ids, attention_mask)
            z = pred.hidden_states[-1][:, 0, :].data.cpu().numpy().astype(np.float32)  # [CLS] embeddings only

        # Normalize vectors by their l2 norm; centers data between 1 and -1 I think
        faiss.normalize_L2(z)
        # z = z / np.linalg.norm(z, ord=2, axis=1)[:, np.newaxis] # Both ways are equivalent

        if is_train:
            index.add(z)
        else:
            scores, _ = index.search(z, num_neighbours)

            if distance == 'Euclidean':
                kth_nn_distances.extend(-1 * scores[:,-1])
            elif distance == 'Cosine':
                # scores are cosine similarity, so higher value means lesser distance
                kth_nn_distances.extend(-1 * (1 - scores[:,-1]))

    return np.array(kth_nn_distances), index



def iterate_data_margin(train_data_loader, id_data_loader, ood_data_loader, model,
                        num_neighbours, margin_type, hidden_size, aggregation, load_embeddings=True):
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
                data = {k: v.to(device) for k, v in data.items()}
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
            aggregated_scores = np.amax(scores, axis=1) # Nearest neighbour
        elif aggregation == 'max':
            aggregated_scores = np.amin(scores, axis=1) # Least similar point
        elif aggregation == 'knn':
            return np.sort(scores, axis=1)[:, -num_neighbours] # np.sort does ascending order

        # Cosine distance = 1 - Cosine Similarity (However this dist breaks triangle inequality).
        # Flip the sign to ensure OOD scores are less than ID
        return -1 * (1 - aggregated_scores)


    if load_embeddings:
        x = np.load(f'../temp_outputs/x.npz')['arr_0']
        index_x = pickle.load(open('../temp_outputs/index_x.pkl', 'rb'))
        y_id = np.load(f'../temp_outputs/y_id.npz')['arr_0']
        y_ood = np.load(f'../temp_outputs/y_ood.npz')['arr_0']
        index_y = pickle.load(open('../temp_outputs/index_y.pkl', 'rb'))
        logger.info('Loaded embeddings.')
    else:
        index_x = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        index_y = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)

        logger.info(f'Calculating indices for train data...')
        x, index_x = get_normalized_embeddings(train_data_loader, index_x, batch_limit=40)
        np.savez_compressed(f'../temp_outputs/x.npz', x)
        pickle.dump(index_x, open('../temp_outputs/index_x.pkl', 'wb'))

        logger.info(f'Calculating indices for ID data...')
        y_id, index_y = get_normalized_embeddings(id_data_loader, index_y, batch_limit=30)
        np.savez_compressed(f'../temp_outputs/y_id.npz', y_id)

        logger.info(f'Calculating indices for OOD data...')
        y_ood, index_y = get_normalized_embeddings(ood_data_loader, index_y, batch_limit=50)
        np.savez_compressed(f'../temp_outputs/y_ood.npz', y_ood)
        pickle.dump(index_y, open('../temp_outputs/index_y.pkl', 'wb'))

    logger.info(f'Calculating margin scores ({margin_type} Margin and {aggregation} aggregation)...')
    scores_total = process_scores(x, np.concatenate([y_id, y_ood]), index_x, index_y)
    in_scores = scores_total[:len(y_id)]
    out_scores = scores_total[len(y_id):]
    assert len(out_scores) == len(y_ood)

    return in_scores, out_scores



def iterate_data_margin_original(train_data_loader, id_data_loader, ood_data_loader, model, num_neighbours, margin_type, hidden_size, aggregation):
    # Ref: https://github.com/facebookresearch/LASER/blob/7e60ad2d58a2ac4f0bb722d5cbfbe2dbf584c561/source/paraphrase.py#L74

    def get_normalized_embeddings(data_loader, batch_limit=3):
        logger.info(f'Starting {len(data_loader)} batches...')
        index = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)

        embeddings = []
        for b, data in enumerate(data_loader):
            if b % 100 == 0:
                logger.info(f'{b} batches processed.')

            if debug_mode and b > batch_limit:
                break

            with torch.no_grad():
                data = {k: v.to(device) for k, v in data.items()}
                input_ids, attention_mask = data['input_ids'], data['attention_mask']
                pred = model(input_ids, attention_mask)
                z = pred.hidden_states[-1][:, 0, :].data.cpu().numpy().astype(np.float32)  # [CLS] embeddings only

            # Normalize vectors by their l2 norm; centers data between 1 and -1 I think
            faiss.normalize_L2(z)
            index.add(z)

            embeddings.extend(z)
        return np.array(embeddings), index


    def process_scores(x, y, index_x, index_y):

        # if aggregation == 'max':
        #     cosine_sim = cosine_sim[:, -1]  # (num_points_in_y, ) #TODO: Confirm if using furthest point (thus -1)
        # elif aggregation == 'avg':
        #     cosine_sim = np.average(cosine_sim, axis=1) # (num_points_in_y, )

        # for batch_num in range(0, y.shape[0], batch_size):
        #     y_batch = y[batch_num: batch_num+batch_size]

        cosine_sim, _ = index_x.search(y, x.shape[0]) # (|Y|, |X|)

        sim_x_to_y, index = index_y.search(x, num_neighbours)  # For each x, find k closest y
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

        # Cosine distance = 1 - Cosine Similarity (However this dist breaks triangle inequality).
        scores = -1 * (1 - scores)

        if aggregation == 'avg':
            return np.average(scores, axis=1)
        elif aggregation == 'max':
            return np.amax(scores, axis=1)
        elif aggregation == 'knn':
            return scores[:, num_neighbours] # TODO: Do I need to resort vals first? Does this even make sense?

    logger.info(f'Calculating indices for train data...')
    x, index_x = get_normalized_embeddings(train_data_loader, batch_limit=40)
    np.savez_compressed(f'../temp_outputs/x.npz', x)
    pickle.dump(index_x, open('../temp_outputs/index_x.pkl', 'wb'))
    x = np.load(f'../temp_outputs/x.npz')['arr_0']
    index_x = pickle.load(open('../temp_outputs/index_x.pkl', 'rb'))

    logger.info(f'Calculating indices for ID data...')
    y_id, index_y_id = get_normalized_embeddings(id_data_loader, batch_limit=30)
    np.savez_compressed(f'../temp_outputs/y_id.npz', y_id)
    pickle.dump(index_y_id, open('../temp_outputs/index_y_id.pkl', 'wb'))
    y_id = np.load(f'../temp_outputs/y_id.npz')['arr_0']
    index_y_id = pickle.load(open('../temp_outputs/index_y_id.pkl', 'rb'))

    logger.info(f'Calculating indices for OOD data...')
    y_ood, index_y_ood = get_normalized_embeddings(ood_data_loader, batch_limit=50)
    np.savez_compressed(f'../temp_outputs/y_ood.npz', y_ood)
    pickle.dump(index_y_ood, open('../temp_outputs/index_y_ood.pkl', 'wb'))
    y_ood = np.load(f'../temp_outputs/y_ood.npz')['arr_0']
    index_y_ood = pickle.load(open('../temp_outputs/index_y_ood.pkl', 'rb'))

    logger.info(f'Calculating ID margin scores ({margin_type} Margin and {aggregation} aggregation)...')
    in_scores = process_scores(x, y_id, index_x, index_y_id)
    np.savez_compressed(f'../temp_outputs/in_scores.npz', in_scores)
    # in_scores = np.load(f'in_scores.npz')['arr_0']

    # ID train -> x
    # ID test -> y1
    # OOD all -> y2

    logger.info(f'Calculating OOD margin scores ({margin_type} Margin and {aggregation} aggregation)...')
    out_scores = process_scores(x, y_ood, index_x, index_y_ood)
    np.savez_compressed(f'../temp_outputs/out_scores.npz', out_scores)
    # out_scores = np.load(f'out_scores.npz')['arr_0']

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
            data = {k: v.to(device) for k, v in data.items()}
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            pred = model(input_ids, attention_mask)
            logits = pred.logits

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

        # debug
        # if b > 500:
        #    break

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []

    for b, data in enumerate(data_loader):
        if b % 100 == 0:
            logger.info(f'{b} batches processed.')

        if debug_mode and b > 3:
            break

        with torch.no_grad():
            data = {k: v.to(device) for k, v in data.items()}
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            pred = model(input_ids, attention_mask)
            logits = pred.logits

            conf = temper * torch.logsumexp(logits / temper, dim=1)  # (N, )
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 10 == 0:
            logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)


def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)  # Applies log on softmax

    for b, data in enumerate(data_loader):
        if b % 1000 == 0:
            print(f'{b} batches processed.')

        if debug_mode and b > 3:
            break

        model.zero_grad()  # Set gradients to 0 for every new batch
        # model.config.output_hidden_states = True

        data = {k: v.to(device) for k,v in data.items()}
        input_ids, attention_mask, labels = data['input_ids'], data['attention_mask'], data['label']
        pred = model(input_ids, attention_mask)#, labels=labels) Not including lables in case we are dealing with varying class datasets.

        # inputs = Variable(input_ids.type(torch.FloatTensor).to(device), requires_grad=True)
        targets = torch.ones((input_ids.shape[0], num_classes)).to(device)
        outputs = pred.logits / temperature

        # The loss for GradNorm is KL(y_hat, target). In the paper, they show this is the same as Cross-Entropy(y_hat, target) - Entropy(target).
        # Entropy(target) is constant so it doesn't show up in any of the backprop calculations. Thus, it's not calculated here either.
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1)) # Cross-Entropy(y_hat, target)
        loss.backward()

        # The loss used to train BERT is cross entropy. However, here we do not use this loss since gradnorm loss is KL btw y_hat and target
        # https://github.com/huggingface/transformers/blob/9aeacb58bab321bc21c24bbdf7a24efdccb1d426/src/transformers/modeling_bert.py#L1359
        # from torch.nn import CrossEntropyLoss
        # loss_fct = CrossEntropyLoss()  # 0.7649
        # loss = loss_fct(pred.logits.view(-1, num_classes), labels.view(-1))

        layer_grad = model.classifier.dense.weight.grad

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def run_eval(model, in_loader, out_loader, logger, args, num_classes, model_config, train_loader):
    # switch to evaluate mode
    model.eval()

    logger.info(f"Running test for {args.score}...")
    logger.info(f'({len(in_loader)} ID batches and {len(out_loader)} OOD batches)')

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)

    elif args.score == 'KNN':
        if args.knn_distance == 'Euclidean':
            index = faiss.IndexFlatL2(model_config.hidden_size)  # 768
        elif args.knn_distance == 'Cosine':
            # Ref: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
            # Ref: https://github.com/facebookresearch/faiss/issues/95
            index = faiss.index_factory(model_config.hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        k = min(args.num_neighbours, args.batch)
        _, index = iterate_data_knn(train_loader, model, index, is_train=True, num_neighbours=k, distance=args.knn_distance)

        logger.info("Processing in-distribution data...")
        in_scores, index = iterate_data_knn(in_loader, model, index, is_train=False, num_neighbours=k, distance=args.knn_distance)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_knn(out_loader, model, index, is_train=False, num_neighbours=k, distance=args.knn_distance)

    elif args.score == 'Margin':

        # cos(A, B) = \frac{A.B}{||A|| ||B||} = \frac{\Sigma A_i B_i}{\sqrt{\Sigma A_i^2} \sqrt{\Sigma B_i^2}}
        # In simpler terms, cosine similarity = inner product (A, B) / L2_norm(A) L2_norm(B)
        # So using FAISS, normalize all vectors by their L2 norm, then find Inner Product.
        # index = faiss.index_factory(model_config.hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        k = min(args.num_neighbours, args.batch)

        logger.info("Processing data...")
        in_scores, out_scores = iterate_data_margin(train_data_loader=train_loader, id_data_loader=in_loader, ood_data_loader=out_loader,
                                                    model=model, hidden_size=model_config.hidden_size, load_embeddings=args.load_embeddings,
                                                    num_neighbours=k, margin_type=args.margin_type, aggregation=args.aggregation)

    elif args.score == 'ODIN': # Needs tuning of temperature, eps
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)

    elif args.score == 'Energy': # Needs tuning of temperature
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)

    elif args.score == 'Mahalanobis': # TODO: Reparameterize
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.to(device) for s in sample_mean]
        precision = [p.to(device) for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).to(device)
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)

    elif args.score == 'GradNorm':
        # TODO: The fact that AUROC is below 0.5 means model is switching classes. Probably a bug.
        #  Check if OOD scores are higher than ID scores.
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes)

    else:
        raise ValueError("Unknown score type {}".format(args.score))

    # Plot histogram of ID vs OOD scores (i.e. frequency of points at each score value)
    np.save(f'in_scores_{args.score}_{args.id_dataset}_to_{args.ood_dataset}.npy', in_scores)
    np.save(f'out_scores_{args.score}_{args.id_dataset}_to_{args.ood_dataset}.npy', out_scores)
    # confs_id = torch.logsumexp(torch.randn(N, C_id), 1).data.cpu().numpy()

    plt.clf()
    plt.hist(in_scores, label='ID data', alpha=0.3, bins='auto')
    plt.hist(out_scores, label='OOD data', alpha=0.3, bins='auto')
    plt.legend()
    plt.grid()
    plt.xlabel('Score')
    plt.ylabel('Freqeuncy')
    # plt.title(f'{args.id_dataset} to {args.ood_dataset} - {args.score} Score')
    # plt.savefig(f'plots/{args.id_dataset} to {args.ood_dataset} - {args.score} Score.png')
    plt.title(f'{args.margin_type} Margin ({args.aggregation} Aggregation, {args.num_neighbours} Neighbours)')
    plt.savefig(f'plots/{args.margin_type} Margin ({args.aggregation} Aggregation, {args.num_neighbours} Neighbours).png')
    plt.show()

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    logger.info(f'In Scores: min={np.min(in_scores)} max={np.max(in_scores)} average={np.average(in_scores)}')
    logger.info(f'Out Scores: min={np.min(out_scores)} max={np.max(out_scores)} average={np.average(out_scores)}')
    logger.info(f'In score samples: {in_scores[:5]}')
    logger.info(f'Out score samples: {out_scores[:5]}')

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

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
    logger = log.setup_logger(args)
    torch.backends.cudnn.benchmark = True
    if args.score == 'GradNorm':
        args.batch = 1
    if debug_mode:
        args.batch = 8

    in_set, out_set, in_loader, out_loader, train_loader = make_id_ood(args, logger)

    logger.info(f"Loading model: {args.model_name} from {MODEL_DIR}.")
    model_config = RobertaConfig.from_pretrained(f'{MODEL_DIR}/{args.model_name}', num_labels=len(in_set.classes),
                    output_hidden_states=True)  # Note: "max_position_embeddings": 514, unlike BERT (512); but embedding size is still 512: https://github.com/pytorch/fairseq/issues/1187
    model = RobertaForSequenceClassification.from_pretrained(f'{MODEL_DIR}/{args.model_name}', config=model_config)
    model = model.to(device)

    if args.score != 'GradNorm':
        model = torch.nn.DataParallel(model)

    start_time = time.time()
    run_eval(model=model, in_loader=in_loader, out_loader=out_loader, logger=logger,
             args=args, num_classes=len(in_set.classes), model_config=model_config, train_loader=train_loader)
    end_time = time.time()
    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    args = Map()
    args.score = 'Margin'         # 'Energy' 'MSP' 'KNN' 'Margin'
    args.id_dataset = 'news-category-modified'
    args.ood_dataset = 'news-category-modified'
    args.model_name = f'roberta_base_{args.id_dataset}' if machine == 'galaxy' else 'roberta_base'

    args.batch = 256
    args.temperature_gradnorm = 1
    args.temperature_energy = 1
    args.name = f'test_GradNorm_iNaturalist'
    args.logdir = 'checkpoints/test_log'
    args.workers = 8

    args.load_embeddings = True
    args.num_neighbours = 50  # For k-NN (Yiyou's paper shows between 10 and 100 is optimal) # Margin paper uses k=4
    args.knn_distance = 'Cosine'  # 'Cosine' 'Euclidean'
    args.margin_type = 'Absolute'    # 'Absolute' 'Distance' 'Ratio'
    args.aggregation = 'knn'      # 'avg' 'max' '1NN' 'knn'

    logger.info(f'Running with args {vars(args)}')
    main(args)
