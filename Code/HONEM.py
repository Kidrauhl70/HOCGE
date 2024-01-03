import numpy as np

class HONEM_Embedding(object):
    """
    Calculates the embedding according to
    [HONEM] Saebi M., Ciampaglia G., Kaplan L., and Chawla N. (2019) 
    'HONEM: Network Embedding Using Higher-Order Patterns in Sequential Data', arXiv:1908.05387

    The instance of class HigherOrderPathGenerator is assumed to contain the rules detected by BuildHON+, see
    [BuildHON+] Xu J., Saebi M., Ribeiro B., Kaplan L., and Chawla N. (2017)
    'Detecting Anomalies in Sequential Data with Higher-order Networks', arXiv:1712.09658
    """
    def __init__(self, graph, dim):
        self.g = graph
        self.dim = dim
    
    def neighborhood_matrix(self, order: int=1, sort=True) -> pd.DataFrame:
        "Calculates the 'v-th order neighborhood matrix' defined in the [HONEM] paper"
        keys = [key for key in self._gen.source_paths if len(key)==order]
        data = [(self.path2str((key[0],)), self.node2str(next_node), prob) 
                for key in keys for _, next_node, prob in self._gen.transition_probs(key)]
        data_df = pd.DataFrame(data, columns=['src','trg','prob'])
        res = data_df.groupby(['src','trg']).mean().unstack(fill_value=0)
        res.columns = [c[1] for c in res.columns] # c[0] == 'prob'
 
        # ensure that neighborhood matrices of different orders have identical indices and columns
        all_source_nodes = { self.path2str((key[0],)) for key in self._gen.source_paths_len1 }
        for v in all_source_nodes.difference(res.index):
            res.loc[v]=0
        all_target_nodes = { self.node2str(node) for node in self._gen.target_nodes }
        for v in all_target_nodes.difference(res.columns):
            res[v]=0

        if sort:
            res = res.loc[self._source._keys_str]
            res = res[self._target._keys_str]
        return res


    
    def train(self):
        # calculate neighborhood matrix
        neighborhood = self.neighborhood_matrix(1)
        if self._gen.max_rule_key_length > 1:
            for order in range(2, self._gen.max_rule_key_length+1): # 公式
                neighborhood = neighborhood + math.exp(1-order) * self.neighborhood_matrix(order, sort=False)
                # neighborhood = neighborhood + self.neighborhood_matrix(order, sort=False)
        # neighborhood = neighborhood / self._gen.max_rule_key_length # 加不加这句似乎没区别
        # sort 确保不同阶数的邻域矩阵具有相同的行和列。
        neighborhood = neighborhood.loc[self._source._keys_str]
        neighborhood = neighborhood[self._target._keys_str]

        source_embedding, target_embedding = self.factor_matrix(neighborhood.values, self._dimension)
        self._source._embedding = source_embedding
        self._target._embedding = target_embedding

    def _decode_raw(self, start: Tuple[Any,...], step = None) -> np.array:
        "Calculate the transition probabilities for start from the embedding. (HONEM does not use the skip-gram model)"
        assert type(start) is tuple and len(start)==1, 'start must be a tuple of length 1'
        return self._source._embedding[self._source._keys_dict[start]] @ self._target._embedding.T
    
    @property
    def config(self):
        "get configuration"
        return dict(init_class=self.__class__.__name__, init_gen=self._gen._id, init_dimension=self._dimension, init_id=self._id)

