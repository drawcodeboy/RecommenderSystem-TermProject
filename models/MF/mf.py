import torch
from torch import nn

class MF(nn.Module):
    '''
    Standard MF (Funk-SVD)
    '''
    def __init__(self,
                 num_users:int = 100, # 유저 수
                 num_items:int = 100, # 아이템 종류 수,
                 latent_dim:int = 100, # latent vector dimension
                 use_bias:bool = False # bias 방식 사용 유무
    ):
        super().__init__()

        self.P = nn.Embedding(num_users, latent_dim)
        self.Q = nn.Embedding(num_items, latent_dim)
        self.b_u = nn.Embedding(num_users, 1) # User bias
        self.b_i = nn.Embedding(num_items, 1) # Item bias
        self.use_bias = use_bias

        # 무슨 문제가 일어날지 모르니.. 일단 초기화
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)

    def forward(self, b_users, b_items):
        '''
        b_users: (B, 1)
        b_items: (B, 1)
        '''

        p = self.P(b_users) # (B, latent_dim)
        p = p.unsqueeze(1) # (B, 1, latent_dim)

        q = self.Q(b_items) # (B, latent_dim)
        q = q.unsqueeze(2) # (B, latent_dim, 1)

        
        pred = torch.bmm(p, q).squeeze(-1) # (B, 1)

        if self.use_bias == False: # Standard MF
            return pred
        else:                      # MF-bias
            b_u = self.b_u(b_users)
            b_i = self.b_i(b_items)
            pred += (b_u + b_i) 
            return pred
        

class NeuralCF(MF):
    def __init__(self,
                 num_users:int = 100, # 유저 수
                 num_items:int = 100, # 아이템 종류 수,
                 latent_dim:int = 100, # latent vector dimension
                 use_bias:bool = False # bias 방식 사용 유무
    ):
        super().__init__(num_users, 
                         num_items, 
                         latent_dim, 
                         use_bias)
        
        self.mlp = nn.Sequential(
            nn.Linear((2*latent_dim) if use_bias == False else (2*(latent_dim+1)), 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.Linear(256, 1)
        )

    def forward(self, b_users, b_items):
        '''
        b_users: (B, 1)
        b_items: (B, 1)
        '''

        p = self.P(b_users) # (B, latent_dim)

        q = self.Q(b_items) # (B, latent_dim)

        if self.use_bias == False:
            concat = torch.cat((p, q), dim=1)
        else:
            b_u = self.b_u(b_users)
            b_i = self.b_i(b_items)
            concat = torch.cat((p, q, b_u, b_i), dim=1)

        pred = self.mlp(concat)
        return pred
    
if __name__ == '__main__':
    model = MF(use_bias=True)
    bz=64
    b_users = torch.randint(0, 100, (bz,))
    b_items = torch.randint(0, 100, (bz,))

    pred = model(b_users, b_items)
    print(pred.shape)


    model = NeuralCF(use_bias=False)
    bz=64
    b_users = torch.randint(0, 100, (bz,))
    b_items = torch.randint(0, 100, (bz,))

    pred = model(b_users, b_items)
    print(pred.shape)