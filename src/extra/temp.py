import numpy as np
import faiss
import time

def test_knn_search(size=10000, gpu_id=None):

    ############ Cosine dist ###############
    dataSetI = [[1, 2, 3], [17, 18, 19]]
    dataSetII = [[4, 5, 6], [1, 2, 3], [17,18,19]]
    # dataSetII = [.1, .2, .3]
    d = len(dataSetI)
    x = np.array(dataSetI).astype(np.float32)
    q = np.array(dataSetII).astype(np.float32)
    index = faiss.index_factory(x.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.ntotal
    faiss.normalize_L2(x)
    index.add(x)
    faiss.normalize_L2(q)
    distance, indices = index.search(q, 3)
    print('Distance by FAISS:{}'.format(distance))

    # To Tally the results check the cosine similarity of the following example
    from scipy import spatial
    result = 1 - spatial.distance.cosine(dataSetI[0], dataSetII[0])
    print('Distance by FAISS:{}'.format(result))

    ############ Euclidean dist ###############

    x = np.random.rand(size, 512)
    x = x.reshape(x.shape[0], -1).astype('float32') # Flatten
    d = x.shape[1]

    tic = time.time()
    if gpu_id is None:
        index = faiss.IndexFlatL2(d)
    else:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id

        flat_config = [cfg]
        resources = [faiss.StandardGpuResources()]
        index = faiss.GpuIndexFlatL2(resources[0], d, flat_config[0])
    index.add(x)
    print('Index built in {} sec'.format(time.time() - tic))

    query_points = np.random.rand(2, d).astype('float32')
    num_neighbours = 5

    nearest_neighbour_distances, nearest_neighbour_indices = index.search(query_points, num_neighbours)
    print('Searched in {} sec'.format(time.time() - tic))
    print(nearest_neighbour_distances.shape)  # (query_points.shape[0], num_neighbours) sorted
    print(nearest_neighbour_indices.shape)    # (query_points.shape[0], num_neighbours)

    ###########################

    # IndexFlatL2 measures the L2 (or Euclidean) distance between all given points between our query vector, and the vectors loaded into the index. It’s simple, very accurate, but not too fast.
    index.is_trained # ideally should be True, otherwise need training
    index.ntotal # Number of points added to index so far



test_knn_search()


# Ref from Yfei: https://github.com/alvinmingwisc/CLIP_OOD/blob/main/utils/detection_util.py#L583
# FAISS documentation: https://faiss.ai/index.html
# FAISS walkthrough: https://www.pinecone.io/learn/faiss-tutorial/

def get_knn_scores_from_clip_img_encoder_id(args, net, train_loader, test_loader):
    '''
    used for KNN score for ID dataset
    '''
    if args.generate:
        ftrain, _ = get_features(args, net, train_loader, dataset = 'ID_train')
        ftest,_ = get_features(args, net, test_loader, dataset = 'ID_test')

    index = faiss.IndexFlatL2(ftrain.shape[1])
    ftrain = ftrain.astype('float32')
    ftest = ftest.astype('float32')
    index.add(ftrain)
    index_bad = index
    D, _ = index_bad.search(ftest, args.K, )
    scores = D[:,-1]
    return scores, index_bad

def get_knn_scores_from_clip_img_encoder_ood(args, net, ood_loader, out_dataset, index_bad):
    '''
    used for KNN score for OOD dataset
    '''
    if args.generate:
        food, _ = get_features(args, net, ood_loader, dataset = out_dataset)
    else:
        with open(os.path.join(args.template_dir, 'all_feat', f'all_feat_{out_dataset}_{args.max_count}_{args.normalize}.npy'), 'rb') as f:
            food =np.load(f)
    food = food.astype('float32')
    D, _ = index_bad.search(food, args.K)
    scores_ood = D[:,-1]
    return scores_ood

