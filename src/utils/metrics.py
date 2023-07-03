import numpy as np


class Metrics:
    def __init__(self, predictions, labels, k):
        """
        Compute several metrics for recommender system
        :param predictions: a list of recommended game_ids
        :param labels: a dictionary {uid: [list of added games]}
        :param k: top-k recommended values
        """
        self.predictions = predictions
        self.labels = labels
        self.uids = list(labels.keys())
        self.k = k

    def map_at_k(self):
        # Mean Average Precision @ K (MAP@K)
        score = 0
        for i in range(len(self.uids)):
            uid = self.uids[i]
            rel_count = 0
            ap = 0
            for j, game_id in enumerate(self.predictions[i][:self.k]):
                if game_id in self.labels[uid]:
                    rel_count += 1
                    ap += (rel_count / (j + 1)) * 1
                else:
                    ap += (rel_count / (j + 1)) * 0
            ap_at_k = 1 / rel_count * ap if rel_count > 0 else 0
            score += ap_at_k
        return score / len(self.uids)

    def recall_at_k(self):
        # Recall @ K
        score = 0
        for i in range(len(self.uids)):
            uid = self.uids[i]
            rel_count = 0
            for j, game_id in enumerate(self.predictions[i][:self.k]):
                if game_id in self.labels[uid]:
                    rel_count += 1
            recall = rel_count / len(self.labels[uid])
            score += recall
        return score / len(self.uids)

    def ndcg_at_k(self):
        # Normalized Discounted Cumulative Gain @ K (NDCG@K)
        score = 0
        for i in range(len(self.uids)):
            uid = self.uids[i]
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(self.k, len(self.labels[uid])))])
            dcg = 0
            for j, game_id in enumerate(self.predictions[i][:self.k]):
                if game_id in self.labels[uid]:
                    dcg += np.reciprocal(np.log2(j + 2))
            score += dcg / idcg
        return score / len(self.uids)


if __name__ == "__main__":
    metrics = Metrics(["1"], [["A", "B", "C", "D", "E", "F", "G"]], {"1": ["B", "X", "D"]}, 6)
    map_score = metrics.map_at_k()
    recall_score = metrics.recall_at_k()
    ndcg_score = metrics.ndcg_at_k()
    print(map_score, recall_score, ndcg_score)
