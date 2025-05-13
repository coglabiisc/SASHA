import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class ABMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, 
                 n_classes = 2, top_k=1, embedding_size=1024, hidden_size=128, dropout_rate=0.25):
        super(ABMIL, self).__init__()
        self.L = embedding_size #1024 # embedding size
        self.D = hidden_size # Hidden size
        self.op = 1
        self.n_classes = n_classes

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

        self.attention_weights = nn.Linear(self.D, self.op)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.op, self.n_classes)
        )

    def forward(self, h, return_features=False, testing=True, temp = 2):
        return_features=False
        # x = x.squeeze(0)

        H = h  # KxL (K X 1024) -- K patches
        # import pdb; pdb.set_trace()
        A_V = self.attention_V(H)  # KxD
        A_U = self.attention_U(H)  # KxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # Kxop
        A = torch.transpose(A, 1, 0)  # opxK
        A_raw = A
        A = F.softmax(A / temp, dim=1)  # softmax over K -- opxK

        M = torch.mm(A, H)  # opXL -- weighted summation

        logits = self.classifier(M) #  1 X 2
        # Y_prob = F.sigmoid(logits)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        Y_prob = F.softmax(logits, dim=-1) # 1 X 2
        Y_hat = torch.argmax(Y_prob, dim=-1) # 1

        result_dict = {
            'A_raw' : A_raw
        }
        return logits, Y_prob, Y_hat, A, result_dict # top_instance, Y_prob, Y_hat, y_probs, results_dict

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    # def relocate(self):
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.classifier.to(device)