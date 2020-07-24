import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
def source_bits(k):
    '''Generates a bit stream of length k bits.
    
    Inputs:
        k: the desired length of the output bitstream
    
    Outputs:
        b: a row vector with k randomly generated bits
    '''
    
    assert k%1 ==0, f'Please input an integer value for the number of transmit bits.'
    
    b = np.random.randint(low=0, high=2, size=(1,int(k)))
    return b
# ----------------------------------------------------------------------------------------------

def get_natural_bit_encoding(B):
    ''' Returns natural integer representation from n-length bit streams.
    
    Inputs:
        B: a (m x n) binary matrix, with m bit sequences of length n  
    
    Outputs:
        s: a (1 x m) where the m-th entry is the decimal representation of the m-th input row 
    '''
    m, n           = B.shape
    weight_matrix  = 2**np.arange(n)[::-1]
    masked_weights = B*weight_matrix
    s              = np.sum(masked_weights ,axis=1, dtype=np.int32).reshape(1,-1)
    
    return s
# ----------------------------------------------------------------------------------------------

def bin2int_mapping(b, M):
    '''Converts bit stream to natural bit mapping of length log2(M).
    inputs:
    # -- b: a row vector with the source bits to be transmitted
    # -- M: the modulation size of the (bits to integer symbols) output space
    # outputs:
    # -- s: a row vector with k/log2(M) integer symbols
    '''
    kb          = b.shape[1]
    spec_eff    = np.log2(M)
    num_packets = kb/spec_eff
    
    assert (spec_eff % 1) == 0, f'Sorry, modulation size must be a power of two; {M} does not meet requirement.'
    assert num_packets % 1 == 0, f'Sorry, number o packets to be transmitted is not an integer multiple of modulation size. \n Please pad the bitstream or change the modulation size accordinly.'
    
    B = b.reshape(int(num_packets),-1)
    s = get_natural_bit_encoding(B)
    
    return s

# ----------------------------------------------------------------------------------------------

def PAM_Alphabet(M, normalize=True):
    ''' Creates a PAM modulation alphabet.
        
        Input:
            M: alphabet size
            normalize: a bool to indicate if the output modulation power should be normalized to 1
    '''
    PAM = np.arange(0, M, 1)
    PAM = 2*PAM - (M-1)
    Es  = (PAM @ PAM.T) / np.size(PAM)
    
    PAM = PAM / np.sqrt(Es) if normalize else PAM

    return PAM

# ----------------------------------------------------------------------------------------------


def get_codebook(path=None, M=None, N=None, LDS=True, normalize=True):
    ''' Loads or creates a multidimensional modulation (a.k.a. codebook).
    
    Inputs:
        path: the path to load the codebook from a file  
        M   : the number of codewords in the codebook    
        N   : the number of dimensions of the codebook
        LDS : a bool to use an N-length repetition code based on an M-ary alphabet
        normalize: sets the output codebook's average codeword power to 1 
    
    Outputs:
        C: a (M x N) matrix where each row is a N-dimensional codeword 
    '''
    
    if path is None and not LDS:
        C = np.random.randn(M,N)
        
    elif path is None and LDS:
        C_1D = PAM_Alphabet(M, normalize=False).reshape(-1,1)
        C    = np.repeat(C_1D,N,axis=1)
    else: 
        # work in progress
        # here we load a codebook from path
        pass
        
    if normalize:
        E = np.sqrt(np.trace(C@C.T)/M)
        C = C/E
        
    return C

# ----------------------------------------------------------------------------------------------


def symbol_to_codeword_map(s, C):
    ''' Selects the codebook's codewords according to the input integer indexes.
    
    Inputs:
        s: a (1 x m) vector of integer values, where each entry corresponds to the correspondent codeword index
        C: a (M x N) matrix where each row is a N-dimensional codeword 
    '''
    
    indexes   = np.reshape(s, newshape=(-1,))
    codewords = C[indexes].reshape(len(list(indexes)), -1)
    
    return codewords

# ----------------------------------------------------------------------------------------------

def iid_rayleigh_channel(X, N0 = 1, fading=True, awgn=True):
    ''' Outputs the received signal after transmiting X over a noisy fading channel. 
    
    Inputs:
        X : an (m x N) array, where the m-th row corresponds to the m-th transmitted symbol
        N0: the noise variance per 2 real dimensions 
        
        fading: if True, channel vector is composed of i.i.d. random fading gains in each dimension
        awgn: if True, adds white noise
        
    Outputs:
        Y: an (m x N) array corresponding to the received signals
        h: an (1 x N) array corresponding to the fading channel vector, same for every channel use
    '''
    
    m, N = X.shape
    
    # block fading channel vector
    if fading:
        h = np.sqrt(1/2)*np.random.randn(1,N)
    else:
        h = np.ones(shape=(1, N))
    
    # channel matrix, repeats the channel vector over m transmissions
    H = np.repeat(h, repeats=m, axis=0)
    
    # noise
    Z = np.zeros(shape=(m,N), dtype=np.float64)
    if awgn:
        Z += np.sqrt(N0/2)*np.random.randn(m, N)
    
    # received signal
    Y = H*X + Z
    
    return Y, h

# ----------------------------------------------------------------------------------------------

def ML_detection(y, HC):
    ''' Maximum Likelihood Detection.
    
    Inputs:
        y : the received signal
        HC: the faded version of the codebook, considering the channel vector 
        
    Outputs:
        hard_symbols: the hard decision on the symbol indexes based on the minimum euclidean distance from y to each codeword in HC
    '''
    pair_dists   = pairwise_distances(X=y, Y=HC)
    hard_symbols = np.argmin(pair_dists,axis=1)
    
    # soft_bits 
    
    return hard_symbols

# ----------------------------------------------------------------------------------------------

def get_min_squared_euclidean_distance(X):
    ''' Calculates the minimum pairwise squared euclidean distance of codebook X.
    
    Inputs:
        X: a (M x N) matrix where each row is a N-dimensional codeword 
    
    Outputs:
        d2_min: the minimum pairwise squared euclidean distance of codebook X
    '''
    
    M         = X.shape[0]
    diag_mask = np.diag(np.ones(shape=(M))*np.inf)
    d2        = pairwise_distances(X, metric='euclidean')**2 + diag_mask
    d2_min    = d2.min()
    
    return d2_min

# ----------------------------------------------------------------------------------------------

def get_L_product_distance(X, eps = 1e-2):
    ''' Calculates the minimum L-product distance of codebook X.
    
    Inputs:
        X  : a (M x N) matrix where each row is a N-dimensional codeword 
        eps: a small threshold precision to consider two values different from each other 
        
    Outputs:
        L_min   : the minimum signal space diversity (a.k.a. the minimum Hamming distance) of the codebook X
        dp_L_min: the minimum (L_min-th)-order pairwise squared euclidean distance of codebook X
    '''
    M, L_min = X.shape
    dp_L_min = np.inf

    for m1 in range(M):
        x1 = X[m1]
        for m2 in range(m1+1,M):
            x2 = X[m2]
            
            diff = np.abs(x1-x2)
            diff = diff[diff>eps]
            
            L = np.size(diff)
            dp_L = np.product(diff)

            if L < L_min:
                L_min = L 
                dp_L_min = dp_L
            elif dp_L < dp_L_min:
                dp_L_min = dp_L

    return (L_min,dp_L_min)

# ----------------------------------------------------------------------------------------------

def get_codebook_figures_of_merit(X, eps=1e-2):
    '''Creates a dictionary of the parameters and figures of merit of codebook X.
    
    Inputs:
        X  : a (M x N) matrix where each row is a N-dimensional codeword 
        eps: a small threshold precision to consider two values different from each other 
        
    Outputs:
        codebook_FoM: a dictionary of items in the format of 'parameter_name':value  
    
    '''
    M, N = X.shape
    de2     = get_min_squared_euclidean_distance(X)
    ssd, dp = get_L_product_distance(X, eps=eps)
    
    Es = np.trace(X@X.T)/X.shape[0]
    
    codebook_FoM = dict({
        'set_cardinality':M,
        'num_of_dimensions':N,
        'min_Euclidean_distance_Squared':de2,
        'min_signal_space_diversity':ssd,
        'min_Product_distance':dp,
        'avg_energy_symbol':Es
                        })
    
    return codebook_FoM

# ----------------------------------------------------------------------------------------------

def plot_multidimensional_modulation(X, figsize=(16,8)):
    '''Creates a dictionary of the parameters and figures of merit of codebook X.
    
    Inputs:
        X  : a (M x N) matrix where each row is a N-dimensional codeword 
        
    Outputs:
        a plot figure is displayed, where 
            each subplot corresponds to a dimension of the codebook and
            the scatter points correspond to the projections the codewords in the current dimension
    '''
    
    xmax = X.max()
    xmin = X.min()
    
    xabs = max(abs(xmax),abs(xmin))
    offset = 0.01*xabs
    
    M, N = X.shape    
    cm = plt.get_cmap('Pastel1')
    colors = [cm(1.*i/M) for i in range(M)]

    plt.figure(figsize=figsize)

    for n in range(N):
        plt.subplot(N,1,n+1)
        
        if n==0:
            plt.title('Codeword projections over each dimension')

        for m in range(M):
            x0, y0 = (X[m,n], 0)

            plt.plot(x0,y0,
                     'o', color=colors[m],
                     alpha=0.85, 
                     markeredgecolor='k',
                     markersize=14,
                     markeredgewidth=.65,)
            
            plt.xlim([-xabs-offset, +xabs+offset])
            
            plt.annotate(s=str(m), xy=(x0,y0),
                        horizontalalignment='center', verticalalignment='center')
            
            plt.ylabel(f'{n}-th dim')

    plt.xlabel('Real dimensions')
    
    plt.show()
    return None