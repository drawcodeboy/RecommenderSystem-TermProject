import torch
import math

def train_one_epoch(model, dataloader, loss_fn, optimizer, task_cfg, device):
    model.train()
    total_loss = []
    
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'modelling':
            optimizer.zero_grad()

            user_id_idx, item_id_idx, rating = data
            user_id_idx, item_id_idx, rating = user_id_idx.to(device), item_id_idx.to(device), rating.to(device)

            pred = model(user_id_idx, item_id_idx)
            loss = loss_fn(pred, rating)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        else:
            raise Exception("Check your task_cfg['object'] configuration")
         
        if task_cfg['object'] == 'modelling':
            print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, "
                  f"RMSE: {sum(total_loss)/len(total_loss):.6f}", end="")
    print()
    
    if task_cfg['object'] == 'modelling':
        return sum(total_loss)/len(total_loss)
    
@torch.no_grad()
def evaluate(model, dataloader, loss_fn, task_cfg, device, K=10, threshold=7.0):
    model.eval()
    
    precision_list = []
    recall_list = []
    ndcg_list = []
    rmse_list = []
    
    # 모든 user별 예측을 모아야 top-K 계산 가능
    user_preds = {}
    user_targets = {}
    
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'modelling':
            user_id_idx, item_id_idx, rating = data
            user_id_idx = user_id_idx.to(device)
            item_id_idx = item_id_idx.to(device)
            rating = rating.to(device)
            
            pred = model(user_id_idx, item_id_idx)
            loss = loss_fn(pred, rating)
            rmse_list.append(loss)
            
            # user별로 예측/실제 rating 저장
            for u, i, r_hat, r_true in zip(user_id_idx, item_id_idx, pred, rating):
                u, i = int(u), int(i)
                
                if u not in user_preds:
                    user_preds[u] = []
                    user_targets[u] = []
                
                user_preds[u].append((i, r_hat.item()))
                user_targets[u].append((i, r_true.item()))

            print(f"\rTest: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()

    for u in user_preds:
        # top-K 예측
        top_k_items = sorted(user_preds[u], key=lambda x: -x[1])[:K]
        top_k_idx = [i for i, _ in top_k_items]
        
        # relevance 변환
        relevance_dict = {i: 1 if r >= threshold else 0 for i, r in user_targets[u]}
        relevant_items = set([i for i, r in user_targets[u] if r >= threshold])
        
        # precision, recall
        num_relevant_in_top_k = sum([relevance_dict[i] for i in top_k_idx])
        precision = num_relevant_in_top_k / K
        recall = num_relevant_in_top_k / len(relevant_items) if len(relevant_items) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        
        # DCG
        dcg = 0.0
        for idx, i in enumerate(top_k_idx):
            rel = relevance_dict[i]
            dcg += (2**rel - 1) / math.log2(idx + 2)  # idx+2 = position+1
        # IDCG
        ideal_rels = sorted([r for _, r in user_targets[u] if r >= threshold], reverse=True)
        idcg = 0.0
        for idx, rel in enumerate(ideal_rels[:K]):
            idcg += (2**1 - 1) / math.log2(idx + 2)  # rel=1
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
    
    precision_at_k = sum(precision_list) / len(precision_list)
    recall_at_k = sum(recall_list) / len(recall_list)
    ndcg_at_k = sum(ndcg_list) / len(ndcg_list)
    rmse = sum(rmse_list) / len(rmse_list)
    
    results = {
        'Precision@K': precision_at_k,
        'Recall@K': recall_at_k,
        'NDCG@K': ndcg_at_k,
        'RMSE': rmse
    }

    return results