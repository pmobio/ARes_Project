import scipy.io as spio
import re
import numpy as np
import torch.nn.functional as F

def machine_translate_using_word(fname, init_embed_file=None, separator='#SPACE#|#COMMA#|#SEMICOLON#|\W+'):
    # some inits
    item_dic = {}
    item_index_id = 0
    # open file
    with open(fname, encoding='utf8') as _file:
        items = [l.split() for l in _file.read().strip().split('\n')]
    print(f'Number of items (entities or relations) read from {fname}: {len(items)}')
    # create item dictionary where key is item
    # and value is the index assigned to that item
    duplicate_froms = []
    duplicate_ids = []
    for item in items:
        if item[0] not in item_dic:
            item_dic[item[0]] = item_index_id
            item_index_id += 1
        else:
            duplicate_froms.append(item_dic[item[0]])
            duplicate_ids.append(item_index_id)
            item_index_id += 1
    print(f'Number of unique items (entities or relations) found: {len(item_dic)}')
    if init_embed_file:
        mat = spio.loadmat(init_embed_file, squeeze_me=True)
        indexed_items = [
            [mat['tree'][i][()][0] - 1]
            if isinstance(mat['tree'][i][()][0], int)
            else [x - 1 for x in mat['tree'][i][()][0]] for i in range(len(mat['tree']))]
        num_words = len(mat['words'])
    else:
        word_index_id = 0
        word_ids = {}
        items_to_words = {}
        # for each item
        for item in items:
            words = []
            # item[0] = molecular_function
            for s in re.split(separator, item[0]):
                words.append(s)
                if s not in word_ids:
                    word_ids[s] = word_index_id
                    word_index_id += 1
            # words = ['molecular', 'function']
            items_to_words[item[0]] = words
        # create list of of length len(item_dic) where all entries are None
        indexed_items = [None] * (len(item_dic)+len(duplicate_ids))
        for key, val in item_dic.items():
            indexed_items[val] = []
            # items_to_words[key] = ['molecular', 'function']
            for s in items_to_words[key]:
                indexed_items[val].append(word_ids[s])
        num_words = len(word_ids)
    for duplicate_from, duplicate_id in zip(duplicate_froms, duplicate_ids):
        indexed_items[duplicate_id] = indexed_items[duplicate_from]
    return indexed_items, num_words, item_dic


def _get_padded_indices_and_weights(indices):

    # find length of each list inside indices
    lens = np.array([len(indices[i]) for i in range(len(indices))])
    mask = np.arange(lens.max()) < lens[:, None]
    padded_indices = np.zeros(mask.shape, dtype=np.int32)
    padded_indices[mask] = np.hstack((indices[:]))
    weights = np.multiply(np.ones(mask.shape, dtype=np.float32) / lens[:, None], mask)
    weights = np.expand_dims(weights, weights.ndim)
    return weights, padded_indices


def lookup_embeddings(padded_entity_indices, word_weights, padded_entity_embedding_weights,):

    entity_embedded_word_ids = F.embedding(
        padded_entity_indices, word_weights
    )  # [batch, seq_len, emb_dim]

    entity_weighted_sum_of_embeddings = entity_embedded_word_ids * padded_entity_embedding_weights

    entity_emb = entity_weighted_sum_of_embeddings.sum(dim=1) # [batch, emb_dim]

    return entity_emb