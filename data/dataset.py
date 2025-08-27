import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np

try:
    from block_dist import extract_input_features
except ImportError:
    try:
        from .block_dist import extract_input_features
    except ImportError:
        from data.block_dist import extract_input_features

class ChainSetDataset(Dataset):
    def __init__(self, csv_path, max_samples=None, contrastive=True):
        self.data = pd.read_csv(csv_path)
        if max_samples:
            self.data = self.data.head(max_samples)
        
        self.contrastive = contrastive
        
        if contrastive:
            self._precompute_features()
    
    def _precompute_features(self):
        self.sample_features = []
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            seq_list = ast.literal_eval(row['seq'])
            seq_list = [s.replace('c','').strip() for s in seq_list]
            features_array = extract_input_features(seq_list)
            
            key_features = {
                'probAA': float(features_array[2]),
                'probBB': float(features_array[3]), 
                'f_A': float(features_array[0]), 
                'mean_block': float(features_array[8]),
                'std_block': float(features_array[9]), 
                'alternation_idx': float(features_array[4] + features_array[5])
            }
            self.sample_features.append(key_features)
    
    def _compute_similarity(self, idx1, idx2):
        feat1 = self.sample_features[idx1]
        feat2 = self.sample_features[idx2]
        
        similarities = []
        
        seq_sim = 1 - abs(feat1['probAA'] - feat2['probAA']) - abs(feat1['probBB'] - feat2['probBB'])
        similarities.append(seq_sim * 2.0)
        
        comp_sim = 1 - abs(feat1['f_A'] - feat2['f_A'])
        similarities.append(comp_sim)
        
        block_sim = 1 - abs(feat1['mean_block'] - feat2['mean_block']) / 10.0  # 归一化
        similarities.append(block_sim)
        
        alt_sim = 1 - abs(feat1['alternation_idx'] - feat2['alternation_idx'])
        similarities.append(alt_sim)
        
        weights = [0.4, 0.2, 0.2, 0.2]
        overall_sim = sum(w * s for w, s in zip(weights, similarities))
        
        return max(0, overall_sim)
    
    def _find_positive_negative(self, idx, pos_threshold=0.8, neg_threshold=0.3):
        if not self.contrastive:
            return None, None
        
        similarities = []
        for other_idx in range(len(self.data)):
            if other_idx != idx:
                sim = self._compute_similarity(idx, other_idx)
                similarities.append((other_idx, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        positive_candidates = [idx for idx, sim in similarities if sim > pos_threshold]
        positive_idx = positive_candidates[0] if positive_candidates else similarities[0][0]
        
        negative_candidates = [idx for idx, sim in similarities if sim < neg_threshold]
        negative_idx = negative_candidates[-1] if negative_candidates else similarities[-1][0]
        
        return positive_idx, negative_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        seq_list = ast.literal_eval(row['seq'])
        seq_list = [s.replace('c','').strip() for s in seq_list]
        
        multi_val_features = ['size','epsAA','epsAB','epsBB','damp','angleKA','angleKB','activationAA','activationBB','activationAB']
        single_val_features = ['Nmono','angleA','angleB','shiftA','shiftB','Temp','epshard']
        x_multi = [row[feat] for feat in multi_val_features]
        x_single = [row[feat] for feat in single_val_features]
        condition_features = torch.tensor(x_multi + x_single, dtype=torch.float32)
        
        input_features = extract_input_features(seq_list)
        
        try:
            from block_dist import extract_target_block_distribution
        except ImportError:
            try:
                from .block_dist import extract_target_block_distribution
            except ImportError:
                from data.block_dist import extract_target_block_distribution
        
        target_block_dist = extract_target_block_distribution(seq_list, max_length=50)
        
        sample_data = {
            'chain_set': seq_list, 
            'condition_features': condition_features, 
            'probAA': torch.tensor(input_features[2], dtype=torch.float32),
            'probBB': torch.tensor(input_features[3], dtype=torch.float32), 
            'target_stats': {
                'f_A': torch.tensor(input_features[0], dtype=torch.float32),
                'mean_block': torch.tensor(input_features[8], dtype=torch.float32),
                'std_block': torch.tensor(input_features[9], dtype=torch.float32),
                'alternation_idx': torch.tensor(input_features[4] + input_features[5], dtype=torch.float32),  # p_AB + p_BA
            }
        }
        
        sample_data['block_dist'] = torch.tensor(target_block_dist, dtype=torch.float32)

        if self.contrastive:
            pos_idx, neg_idx = self._find_positive_negative(idx)
            sample_data['positive_idx'] = pos_idx
            sample_data['negative_idx'] = neg_idx
        
        return sample_data

def encode_single_chain(chain, max_length=500):

    char_to_idx = {'A': 1, 'B': 2, 'PAD': 0}
    
    if len(chain) > max_length:
        chain = chain[:max_length]
    
    encoded = [char_to_idx.get(char, 0) for char in chain]
    
    while len(encoded) < max_length:
        encoded.append(char_to_idx['PAD'])
    
    return torch.tensor(encoded, dtype=torch.long)

def collate_fn_set_transformer(batch):

    batch_size = len(batch)
    
    all_chain_sets = [item['chain_set'] for item in batch]
    condition_features = torch.stack([item['condition_features'] for item in batch])
    
    max_set_size = max(len(chain_set) for chain_set in all_chain_sets)
    max_chain_length = 500
    
    batch_chain_sets = torch.zeros(batch_size, max_set_size, max_chain_length, dtype=torch.long)
    set_masks = torch.zeros(batch_size, max_set_size, dtype=torch.bool) 
    
    for i, chain_set in enumerate(all_chain_sets):
        for j, chain in enumerate(chain_set[:max_set_size]):
            batch_chain_sets[i, j] = encode_single_chain(chain, max_chain_length)
            set_masks[i, j] = True 
    
    probAAs = torch.stack([item['probAA'] for item in batch])
    probBBs = torch.stack([item['probBB'] for item in batch])
    block_dists = torch.stack([item['block_dist'] for item in batch])

    target_stats = {}
    for key in ['f_A', 'mean_block', 'std_block', 'alternation_idx']:
        target_stats[key] = torch.stack([item['target_stats'][key] for item in batch])
    
    result = {
        'chain_sets': batch_chain_sets,      # [B, max_set_size, max_chain_length] 
        'set_masks': set_masks,              # [B, max_set_size] - True表示真实chain
        'condition_features': condition_features,
        'probAAs': probAAs,
        'probBBs': probBBs,
        'block_dists': block_dists,
        'target_stats': target_stats
    }

    if 'positive_idx' in batch[0]:
        result['positive_indices'] = torch.tensor([item['positive_idx'] for item in batch])
        result['negative_indices'] = torch.tensor([item['negative_idx'] for item in batch])
    
    return result

if __name__ == "__main__":
    dataset = ChainSetDataset('~/copolymer/data/copolymer.csv', max_samples=1000, contrastive=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True,
        collate_fn=collate_fn_set_transformer,
        num_workers=0  
    )
    
    for batch in dataloader:
        print(f"Chain sets shape: {batch['chain_sets'].shape}")
        print(f"Set masks shape: {batch['set_masks'].shape}")
        print(f"Block dists shape: {batch['block_dists'].shape}")
        if 'positive_indices' in batch:
            print(f"Positive indices: {batch['positive_indices']}")
        break