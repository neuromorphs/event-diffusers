import torch


def embed(model, labels, device):
    """Embed the labels through a simple model.

    Arguments:
        model: A torch.nn.Module
        labels: A tensor of shape (batch_size, num_labels)

    Returns:
        A tensor of shape (batch_size, embedding_size)
    """
    model = model.to(device)
    emb = model(labels.to(device)).unsqueeze(1)
    return emb


class EmbedFC(torch.nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        generic one layer FC NN for embedding things  
        """
        self.input_dim = input_dim
        layers = [
            torch.nn.Linear(input_dim, emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, emb_dim),
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x.type(torch.float32))
