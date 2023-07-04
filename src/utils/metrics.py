import numpy as np


class Metrics:
    def __init__(self, predictions, labels):
        """
        Compute several metrics for recommender system
        :param predictions: a list of recommended game_ids
        :param labels: a dictionary {uid: [list of added games]}
        """
        self.predictions = predictions
        self.labels = labels
        self.uids = list(labels.keys())

    def evaluate(self, topk):
        """
        Evaluation
        :param topk: int or list of ints
        :return:
        """
        if type(topk) == int:
            map = self.map_at_k(topk)
            recall = self.recall_at_k(topk)
            ndcg = self.ndcg_at_k(topk)
            print(f"MAP@{topk}: {map}\nRecall@{topk}: {recall}\nNDCG@{topk}: {ndcg}")
            return (map, recall, ndcg)
        else:
            all_metrics = list()
            for k_val in topk:
                map = self.map_at_k(k_val)
                recall = self.recall_at_k(k_val)
                ndcg = self.ndcg_at_k(k_val)
                print(f"MAP@{k_val}: {map}\nRecall@{k_val}: {recall}\nNDCG@{k_val}: {ndcg}")
                print("============")
                all_metrics.append((map, recall, ndcg))
            return all_metrics

    def map_at_k(self, k):
        # Mean Average Precision @ K (MAP@K)
        score = 0
        for i in range(len(self.uids)):
            uid = self.uids[i]
            rel_count = 0
            ap = 0
            for j, game_id in enumerate(self.predictions[i][:k]):
                if game_id in self.labels[uid]:
                    rel_count += 1
                    ap += (rel_count / (j + 1)) * 1
                else:
                    ap += (rel_count / (j + 1)) * 0
            ap_at_k = 1 / rel_count * ap if rel_count > 0 else 0
            score += ap_at_k
        return score / len(self.uids)

    def recall_at_k(self, k):
        # Recall @ K
        score = 0
        for i in range(len(self.uids)):
            uid = self.uids[i]
            rel_count = 0
            for j, game_id in enumerate(self.predictions[i][:k]):
                if game_id in self.labels[uid]:
                    rel_count += 1
            recall = rel_count / len(self.labels[uid])
            score += recall
        return score / len(self.uids)

    def ndcg_at_k(self, k):
        # Normalized Discounted Cumulative Gain @ K (NDCG@K)
        score = 0
        for i in range(len(self.uids)):
            uid = self.uids[i]
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(k, len(self.labels[uid])))])
            dcg = 0
            for j, game_id in enumerate(self.predictions[i][:k]):
                if game_id in self.labels[uid]:
                    dcg += np.reciprocal(np.log2(j + 2))
            score += dcg / idcg
        return score / len(self.uids)


if __name__ == "__main__":
    metrics = Metrics([["A", "B", "C", "D", "E", "F", "G"]], {"1": ["B", "X", "D"]})
    results = metrics.evaluate([1, 3, 6])
