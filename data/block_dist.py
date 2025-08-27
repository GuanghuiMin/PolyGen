import numpy as np
from collections import Counter

def extract_input_features(polymer_chains, normalize=True):
    """
    Extract ONLY input features for prediction (excluding target block distribution)
    Goal: predict block_dist from p_AA, p_BB and other non-block-distribution features
    """
    
    if not polymer_chains:
        return np.zeros(41)  # 41 features
    
    total_chains = len(polymer_chains)
    total_monomers = sum(len(chain) for chain in polymer_chains)
    all_monomers = ''.join(polymer_chains)
    
    # Extract block data (for type-specific features only, not for histogram)
    all_block_lengths = []
    typed_blocks = {'A': [], 'B': []}
    
    for chain in polymer_chains:
        if chain:
            current_length = 1
            current_monomer = chain[0]
            
            for i in range(1, len(chain)):
                if chain[i] == current_monomer:
                    current_length += 1
                else:
                    all_block_lengths.append(current_length)
                    typed_blocks[current_monomer].append(current_length)
                    current_monomer = chain[i]
                    current_length = 1
            
            all_block_lengths.append(current_length)
            typed_blocks[current_monomer].append(current_length)
    
    features = []
    
    # 1. Composition features (2 dimensions)
    f_A = all_monomers.count('A') / len(all_monomers) if all_monomers else 0
    f_B = 1 - f_A
    features.extend([f_A, f_B])
    
    # 2. Sequence features (6 dimensions)
    all_pairs = []
    all_triplets = []
    
    for chain in polymer_chains:
        if len(chain) >= 2:
            chain_pairs = [chain[i:i+2] for i in range(len(chain)-1)]
            all_pairs.extend(chain_pairs)
        if len(chain) >= 3:
            chain_triplets = [chain[i:i+3] for i in range(len(chain)-2)]
            all_triplets.extend(chain_triplets)
    
    pair_counts = Counter(all_pairs)
    triplet_counts = Counter(all_triplets)
    
    total_pairs = len(all_pairs)
    total_triplets = len(all_triplets)
    
    # These are your KEY INPUT FEATURES
    p_AA = pair_counts['AA'] / total_pairs if total_pairs > 0 else 0
    p_BB = pair_counts['BB'] / total_pairs if total_pairs > 0 else 0  
    p_AB = pair_counts['AB'] / total_pairs if total_pairs > 0 else 0
    p_BA = pair_counts['BA'] / total_pairs if total_pairs > 0 else 0
    
    p_AAA = triplet_counts['AAA'] / total_triplets if total_triplets > 0 else 0
    p_BBB = triplet_counts['BBB'] / total_triplets if total_triplets > 0 else 0
    
    features.extend([p_AA, p_BB, p_AB, p_BA, p_AAA, p_BBB])
    
    # 3. Block statistics (10 dimensions) ✅ - But NO histogram
    if all_block_lengths:
        block_array = np.array(all_block_lengths)
        
        features.extend([
            np.mean(block_array),  # These stats are OK as they're derived from p_AA, p_BB
            np.std(block_array),   
            np.min(block_array),   
            np.max(block_array),
        ])
        
        percentiles = np.percentile(block_array, [25, 50, 75, 90])
        features.extend(percentiles.tolist())
        
        mean = np.mean(block_array)
        std = np.std(block_array)
        if std > 0:
            skew = np.mean(((block_array - mean) / std) ** 3)  
            kurt = np.mean(((block_array - mean) / std) ** 4) - 3  
        else:
            skew = kurt = 0
        features.extend([skew, kurt])
        
    else:
        features.extend([0] * 10)
    
    for monomer_type in ['A', 'B']:
        type_blocks = typed_blocks[monomer_type]
        if type_blocks:
            type_array = np.array(type_blocks)
            features.extend([
                np.mean(type_array),
                np.std(type_array),
                len(type_array),  
                np.max(type_array),
                len(type_array) / len(all_block_lengths) if all_block_lengths else 0
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
    
    chain_lengths = [len(chain) for chain in polymer_chains]
    if chain_lengths:
        chain_array = np.array(chain_lengths)
        features.extend([
            np.mean(chain_array),     
            np.std(chain_array),      
            len(chain_lengths),       
            np.sum(chain_array),      
            len(all_block_lengths) / len(chain_lengths) if chain_lengths else 0,
            np.std(chain_array) / np.mean(chain_array) if np.mean(chain_array) > 0 else 0  
        ])
    else:
        features.extend([0] * 6)
    
    # 7. Reactivity ratio estimation (7 dimensions) ✅ - Mayo-Lewis theory
    if p_AB > 0 and f_A > 0 and f_B > 0:
        r_A_est = (p_AA / (p_AB/2)) * (f_B / f_A) if p_AB > 0 else 1.0
        r_B_est = (p_BB / (p_AB/2)) * (f_A / f_B) if p_AB > 0 else 1.0
    else:
        r_A_est = r_B_est = 1.0
    
    # Constrain to reasonable range
    r_A_est = max(0.01, min(10.0, r_A_est))
    r_B_est = max(0.01, min(10.0, r_B_est))
    
    features.extend([
        r_A_est,
        r_B_est,
        r_A_est * r_B_est,  
        np.log(r_A_est),    
        np.log(r_B_est),
        p_AB / (p_AA + p_BB + 1e-8),  # alternation index
        1.0 if r_A_est * r_B_est < 1 else 0.0  # alternating tendency flag
    ])
    
    # Convert to numpy array - Should be exactly 41 features now
    feature_vector = np.array(features, dtype=np.float32)
    
    # Numerical stability
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=10.0, neginf=-10.0)
    
    if normalize:
        feature_vector = normalize_input_features(feature_vector)
    
    assert len(feature_vector) == 41, f"Expected 41 features, got {len(feature_vector)}"
    
    return feature_vector

def normalize_input_features(features):
    """Normalize the 41 input features"""
    
    # Define normalization parameters for 41 features
    feature_means = np.array([
        # Composition (2)
        0.5, 0.5,
        # Sequence (6) 
        0.3, 0.3, 0.2, 0.2, 0.1, 0.1,
        # Block stats (10)
        5.0, 3.0, 1.0, 15.0, 2.0, 5.0, 8.0, 12.0, 0.5, 1.0,
        # Type A (5)
        5.0, 3.0, 50.0, 15.0, 0.5,
        # Type B (5)  
        5.0, 3.0, 50.0, 15.0, 0.5,
        # Chain structure (6)
        20.0, 10.0, 100.0, 2000.0, 10.0, 0.5,
        # Reactivity (7)
        1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.3
    ], dtype=np.float32)
    
    feature_stds = np.array([
        # Composition (2)
        0.2, 0.2,
        # Sequence (6)
        0.15, 0.15, 0.1, 0.1, 0.05, 0.05,
        # Block stats (10)
        3.0, 2.0, 1.0, 10.0, 2.0, 3.0, 5.0, 8.0, 0.5, 1.0,
        # Type A (5)
        3.0, 2.0, 30.0, 10.0, 0.3,
        # Type B (5)
        3.0, 2.0, 30.0, 10.0, 0.3,
        # Chain structure (6)
        15.0, 8.0, 80.0, 1500.0, 5.0, 0.3,
        # Reactivity (7)
        2.0, 2.0, 3.0, 1.0, 1.0, 0.3, 0.4
    ], dtype=np.float32)
    
    assert len(features) == len(feature_means), f"Feature length mismatch: {len(features)} vs {len(feature_means)}"
    
    normalized = (features - feature_means) / feature_stds
    normalized = np.clip(normalized, -5.0, 5.0)
    
    return normalized

def extract_target_block_distribution(polymer_chains, max_length=20):
    """Extract the TARGET block distribution that we want to predict"""
    
    all_block_lengths = []
    
    for chain in polymer_chains:
        if chain:
            current_length = 1
            current_monomer = chain[0]
            
            for i in range(1, len(chain)):
                if chain[i] == current_monomer:
                    current_length += 1
                else:
                    all_block_lengths.append(current_length)
                    current_monomer = chain[i]
                    current_length = 1
            
            all_block_lengths.append(current_length)
    
    # Build probability distribution
    counter = Counter(all_block_lengths)
    lengths = np.arange(1, max_length + 1)
    probs = np.array([counter.get(length, 0) for length in lengths], dtype=np.float32)
    
    # Normalize
    if probs.sum() > 0:
        probs = probs / probs.sum()
    
    return probs

def mayo_lewis_from_sequence(polymer_chains, max_length=50):
    """
    Calculate theoretical block distribution using Mayo-Lewis theory
    directly from polymer sequences by extracting statistics
    
    Args:
        polymer_chains: List of polymer sequences (e.g., ['AABBA', 'BBAAA'])
        max_length: Maximum block length to compute (default: 50)
    
    Returns:
        np.array: Theoretical block length distribution according to Mayo-Lewis theory
    """
    if not polymer_chains:
        return np.zeros(max_length, dtype=np.float32)
    
    # Extract key statistics from sequences
    all_monomers = ''.join(polymer_chains)
    
    # Calculate composition
    f_A = all_monomers.count('A') / len(all_monomers) if all_monomers else 0
    f_B = 1 - f_A
    
    # Calculate pair transition probabilities
    all_pairs = []
    for chain in polymer_chains:
        if len(chain) >= 2:
            chain_pairs = [chain[i:i+2] for i in range(len(chain)-1)]
            all_pairs.extend(chain_pairs)
    
    if not all_pairs:
        # No pairs available, return uniform distribution
        dist = np.ones(max_length, dtype=np.float32)
        return dist / dist.sum()
    
    pair_counts = Counter(all_pairs)
    total_pairs = len(all_pairs)
    
    p_AA = pair_counts['AA'] / total_pairs
    p_BB = pair_counts['BB'] / total_pairs
    p_AB = pair_counts['AB'] / total_pairs
    p_BA = pair_counts['BA'] / total_pairs
    
    # Use the existing theoretical_block_prediction function
    return theoretical_block_prediction(p_AA, p_BB, f_A, max_length)

def theoretical_block_prediction(p_AA, p_BB, f_A, max_length=20):
    """
    Predict block distribution using Mayo-Lewis theory
    This shows what's theoretically possible from just p_AA, p_BB, f_A
    """
    
    f_B = 1 - f_A
    
    # Estimate reactivity ratios
    p_AB = 1 - p_AA - p_BB  # Assuming p_AB + p_BA = 1 - p_AA - p_BB
    
    if p_AB > 0 and f_A > 0 and f_B > 0:
        r_A = (p_AA / (p_AB/2)) * (f_B / f_A)
        r_B = (p_BB / (p_AB/2)) * (f_A / f_B)
    else:
        r_A = r_B = 1.0
    
    r_A = max(0.01, min(10.0, r_A))
    r_B = max(0.01, min(10.0, r_B))
    
    # Mayo-Lewis block length distribution
    lengths = np.arange(1, max_length + 1)
    
    # Continuation probabilities
    if r_A * f_A + f_B > 0:
        p_A_continue = r_A * f_A / (r_A * f_A + f_B)
    else:
        p_A_continue = 0
        
    if r_B * f_B + f_A > 0:
        p_B_continue = r_B * f_B / (r_B * f_B + f_A)
    else:
        p_B_continue = 0
    
    # Geometric distributions for each type
    if p_A_continue > 0:
        probs_A = (1 - p_A_continue) * (p_A_continue ** (lengths - 1))
    else:
        probs_A = np.zeros_like(lengths, dtype=float)
        probs_A[0] = 1.0
    
    if p_B_continue > 0:
        probs_B = (1 - p_B_continue) * (p_B_continue ** (lengths - 1))
    else:
        probs_B = np.zeros_like(lengths, dtype=float)
        probs_B[0] = 1.0
    
    # Weighted combination
    combined_probs = f_A * probs_A + f_B * probs_B
    
    if combined_probs.sum() > 0:
        combined_probs = combined_probs / combined_probs.sum()
    
    return combined_probs.astype(np.float32)

if __name__ == '__main__':
    # Test with sample data from CSV
    import pandas as pd
    import ast
    
    # Try to load the first sequence from the CSV file
    try:
        df = pd.read_csv('./data/copolymer.csv')
        # Get the seq column which contains a list of sequences as string
        seq_str = df.iloc[0]['seq']
        # Parse the string representation of list to actual list
        sequences = ast.literal_eval(seq_str)
        test_seq = sequences[0]  # Get first sequence
        print(f"Loaded {len(sequences)} sequences from CSV")
        print(f"Testing with first sequence: {test_seq[:50]}...")  # Show first 50 chars
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Fallback test sequence
        test_seq = 'AABABBAABABBAABBAABAABB'
        sequences = [test_seq]
        print(f"Using fallback test sequence: {test_seq}")
    
    # Test original functions
    print("\n--- Testing Input Features ---")
    input_features = extract_input_features(sequences)
    print(f"Input features shape: {input_features.shape}")
    print(f"Input features: {input_features[:5]}")  # Show first 5 features
    
    print("\n--- Testing Block Distribution ---")
    block_dist = extract_target_block_distribution(sequences)
    print(f"Block distribution shape: {block_dist.shape}")
    print(f"Block distribution: {block_dist[:10]}")  # Show first 10 values
    
    # Test the new Mayo-Lewis function with sequence data
    print("\n--- Testing Mayo-Lewis from Sequence ---")
    theoretical_dist = mayo_lewis_from_sequence(sequences[:5], max_length=len(block_dist))  # Use first 5 sequences
    print(f"Theoretical block distribution shape: {theoretical_dist.shape}")
    print(f"Theoretical block distribution: {theoretical_dist[:10]}")  # Show first 10 values
    
    # Compare observed vs theoretical
    print("\n--- Comparison: Observed vs Theoretical ---")
    for i in range(min(10, len(block_dist), len(theoretical_dist))):
        print(f"Block length {i+1}: Observed={block_dist[i]:.4f}, Theoretical={theoretical_dist[i]:.4f}")
    
    # Calculate correlation
    try:
        from scipy.stats import pearsonr
        if len(block_dist) == len(theoretical_dist):
            correlation, p_value = pearsonr(block_dist, theoretical_dist)
            print(f"\nCorrelation between observed and theoretical: {correlation:.4f} (p={p_value:.4f})")
    except ImportError:
        print("\nScipy not available, skipping correlation calculation")
    
    print("\n--- Testing Complete ---")
