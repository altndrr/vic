import torch


class NearestNeighboursClassifier(torch.nn.Module):
    """Nearest neighbours classifier.

    It computes the similarity between the query and the supports using the
    cosine similarity and then applies a softmax to obtain the logits.

    Args:
        scale (float): Scale for the logits of the query. Defaults to 1.0.
        tau (float): Temperature for the softmax. Defaults to 1.0.
    """

    def __init__(self, scale: float = 1.0, tau: float = 1.0):
        super().__init__()
        self.scale = scale
        self.tau = tau

    def forward(self, query: torch.Tensor, supports: torch.Tensor):
        query = query / query.norm(dim=-1, keepdim=True)
        supports = supports / supports.norm(dim=-1, keepdim=True)

        if supports.dim() == 2:
            supports = supports.unsqueeze(0)

        Q, _ = query.shape
        N, C, _ = supports.shape

        supports = supports.mean(dim=0)
        supports = supports / supports.norm(dim=-1, keepdim=True)
        similarity = self.scale * query @ supports.T
        similarity = similarity / self.tau if self.tau != 1.0 else similarity
        logits = similarity.softmax(dim=-1)

        return logits
