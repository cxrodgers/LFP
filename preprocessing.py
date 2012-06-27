"""Methods for loading and preprocessing LFP data from ns5 files"""
from ns5_process import ns5
from ns5_process.myutils import printnow
import numpy as np
import scipy.signal

def load_as_array(datafile, channel_groups, starts=None, downsampled_chunksize=128,
    oversize=1.2, downsample_ratio=64, verbose=False, n_chunks=None):
    """Loads LFP from ns5 file, downsamples, returns as array.
    
    datafile : ns5 file
    channel_groups : list of lists, containing channel ids to grab.
        Example 1: [[1,2,3], [5,6,7,8]]. Will return two groups, the average
        of ch1 + ch2 + ch3, and the average of chs 5-8
        Example 2: [[1], [2]]. Will return two groups, one for ch1 and one for
        ch2.    
    starts : integer locations in samples where chunks should begin
        If None, will return non-overlapping chunks spanning the file
    downsampled_chunksize : size of returned chunks. For the purpose of this
        method there is no benefit to choosing a power of 2 (because see below).
    oversize : amount larger than returned chunk to actually grab, to avoid
        edge effects. The size of the larger chunk is always a power of 2
        for efficiency.
    downsample_ratio : amount to downsample data before returning
    verbose : display error messages    
    n_chunks : truncate `starts` to this length, mostly useful for
        benchmarking
    
    Returns: values, t
        values : array of shape (n_chunks, n_groups, downsampled_chunksize)
        t : array of same shape, containing times corresponding to each point
            dim2 is redundant since all groups are timelocked.
    """
    # Load header
    l = ns5.Loader()
    l.load_file(datafile)
    n_samples = l.header.n_samples
    fs = l.header.f_samp
    
    # Set up chunk parameters
    # We will actually do the operations on a slightly larger, potentially
    # overlapping, chunk. Then we slice out just the target area.
    downsample_ratio = int(downsample_ratio)
    oversize = float(oversize)
    downsampled_chunksize = int(downsampled_chunksize)
    chunksize = downsample_ratio * downsampled_chunksize
    bigchunksize = 2**(int(np.ceil(np.log2(oversize * chunksize))))

    # Define how much to grab on either side of desired chunk
    # bigchunk = data[start - prechunk:start - prechunk + bigchunksize]
    prechunk = (bigchunksize - chunksize) / 2

    # Define where to get the chunks from
    if starts is None:
        starts = np.arange(prechunk, n_samples + prechunk - bigchunksize + 1, 
            chunksize, dtype=np.int)
    if n_chunks is not None:
        starts = starts[:n_chunks]

    # Calculate size of everything post downsampling
    post_bigchunksize = bigchunksize / downsample_ratio
    post_chunksize = chunksize / downsample_ratio
    post_prechunk = prechunk / downsample_ratio

    # Call to underlying implementation
    # They all return the same values but do the operations in different orders
    # Method 4 seems to be the best. It avoids the worst step, which is
    # creation of intermediate arrays, by using get_chunk instead of
    # get_chunk_by_channel. It also downsamples before meaning, to decrease
    # the number of data points.
    # Methods 1-3 use get_chunk_by_channel and 3 is the best of them because
    # it downsamples before meaning and processes groups separately.
    # There appear to be minimal performance gains from vectorizing the
    # resampling operation, at least not if we include the cost of creating
    # the array.
    res, rest = _load_as_array6(l, channel_groups, starts, bigchunksize,
        prechunk, fs, post_prechunk, post_chunksize, post_bigchunksize, verbose)
    
    # Okay one final improvement - method 6 does the resampling all at once
    # on the original chunk to benefit from any possible vectorization
    # without creating a new array. Also it slices out the chunk before
    # meaning to avoid unnecessary meaning.
    #
    # Note that for smaller downsample_ratios or especially non-integer
    # ratios it would probably be beneficial to mean before downsampling.
    # In that case method 5 is probably the best, though it could still
    # benefit from some of the improvements in method 6.
    
    return res, rest


# Different implementations of the inner loop
def _load_as_array1(l, channel_groups, starts, bigchunksize, prechunk, fs, 
    post_prechunk, post_chunksize, post_bigchunksize, verbose):
    """Process groups separately, mean first, then downsample"""
    # Set up return values
    res_l, rest_l = [], []
    for start in starts:
        # Grab current chunk
        if verbose:
            printnow("loading chunk starting at %d" % start)
        raw = l.get_chunk_by_channel(start=start-prechunk, n_samples=bigchunksize)
        t = np.arange(start-prechunk, start - prechunk + bigchunksize) / float(fs) 

        # Now process one group at a time
        # This might be faster if mean and downsample all groups at once?
        downsampled_l, t_l = [], []
        for group in channel_groups:
            # Mean and downsample
            meaned = np.mean([raw[ch] for ch in group], axis=0)        
            downsampled, new_t = scipy.signal.resample(meaned, 
                post_bigchunksize, t=t)
            
            # Slice out just desired chunk
            # If you grab one more sample here on the ends, you can check how
            # well the overlap is working between chunks
            downsampled = downsampled[post_prechunk:post_prechunk+post_chunksize]
            new_t = new_t[post_prechunk:post_prechunk+post_chunksize]
            
            # Append to result
            downsampled_l.append(downsampled)
            t_l.append(new_t)
        res_l.append(downsampled_l)
        rest_l.append(t_l)
    res = np.array(res_l)
    rest = np.array(rest_l)

    return res, rest


def _load_as_array2(l, channel_groups, starts, bigchunksize, prechunk, fs, 
    post_prechunk, post_chunksize, post_bigchunksize, verbose):
    """Process groups together, mean first, then downsample"""
    # Set up return values
    res_l, rest_l = [], []
    for start in starts:
        # Grab current chunk
        if verbose:
            printnow("loading chunk starting at %d" % start)
        raw = l.get_chunk_by_channel(start=start-prechunk, n_samples=bigchunksize)
        t = np.arange(start-prechunk, start - prechunk + bigchunksize) / float(fs) 

        # Now process all groups at once
        all = np.array([np.mean([raw[ch] for ch in group], axis=0) 
            for group in channel_groups])
        dsall, new_t = scipy.signal.resample(all, post_bigchunksize, t=t, axis=1)
        new_t = np.tile(new_t, (len(channel_groups), 1))
        
        # Slice out just desired chunk
        # If you grab one more sample here on the ends, you can check how
        # well the overlap is working between chunks
        dsall = dsall[:, post_prechunk:post_prechunk+post_chunksize]
        new_t = new_t[:, post_prechunk:post_prechunk+post_chunksize]
        
        res_l.append(dsall)
        rest_l.append(new_t)
    res = np.array(res_l)
    rest = np.array(rest_l)

    return res, rest


def _load_as_array3(l, channel_groups, starts, bigchunksize, prechunk, fs, 
    post_prechunk, post_chunksize, post_bigchunksize, verbose):
    """Process groups separately, downsample first, then mean"""
    # Set up return values
    res_l, rest_l = [], []
    for start in starts:
        # Grab current chunk
        if verbose:
            printnow("loading chunk starting at %d" % start)
        raw = l.get_chunk_by_channel(start=start-prechunk, n_samples=bigchunksize)
        t = np.arange(start-prechunk, start - prechunk + bigchunksize) / float(fs) 

        # Now process one group at a time
        downsampled_l, t_l = [], []
        for group in channel_groups:
            # Mean and downsample
            rawgroup = np.asarray([raw[ch] for ch in group])        
            downsampled, new_t = scipy.signal.resample(rawgroup, 
                post_bigchunksize, t=t, axis=1)
            downsampled = np.mean(downsampled, axis=0)
            
            # Slice out just desired chunk
            # If you grab one more sample here on the ends, you can check how
            # well the overlap is working between chunks
            downsampled = downsampled[post_prechunk:post_prechunk+post_chunksize]
            new_t = new_t[post_prechunk:post_prechunk+post_chunksize]
            
            # Append to result
            downsampled_l.append(downsampled)
            t_l.append(new_t)
        res_l.append(downsampled_l)
        rest_l.append(t_l)
    res = np.array(res_l)
    rest = np.array(rest_l)

    return res, rest


def _load_as_array4(l, channel_groups, starts, bigchunksize, prechunk, fs, 
    post_prechunk, post_chunksize, post_bigchunksize, verbose):
    """Avoid using intermediate dict object, downsample first"""
    # Set up return values
    res_l, rest_l = [], []
    for start in starts:
        # Grab current chunk
        if verbose:
            printnow("loading chunk starting at %d" % start)
        raw = l.get_chunk(start=start-prechunk, n_samples=bigchunksize)
        t = np.arange(start-prechunk, start - prechunk + bigchunksize) / float(fs) 

        # Now process one group at a time
        downsampled_l, t_l = [], []
        for group in channel_groups:
            # Column indexes into raw
            igroup = [l.header.Channel_ID.index(ch) for ch in group]
            
            # Mean and downsample
            #rawgroup = np.asarray([raw[ch] for ch in group])        
            downsampled, new_t = scipy.signal.resample(raw[:, igroup], post_bigchunksize, t=t, axis=0)
            downsampled = np.mean(downsampled, axis=1)
            
            # Slice out just desired chunk
            # If you grab one more sample here on the ends, you can check how
            # well the overlap is working between chunks
            downsampled = downsampled[post_prechunk:post_prechunk+post_chunksize]
            new_t = new_t[post_prechunk:post_prechunk+post_chunksize]
            
            # Append to result
            downsampled_l.append(downsampled)
            t_l.append(new_t)
        res_l.append(downsampled_l)
        rest_l.append(t_l)
    res = np.array(res_l)
    rest = np.array(rest_l)

    return res, rest



def _load_as_array5(l, channel_groups, starts, bigchunksize, prechunk, fs, 
    post_prechunk, post_chunksize, post_bigchunksize, verbose):
    """Avoid using intermediate dict object, mean first"""
    # Set up return values
    res_l, rest_l = [], []
    for start in starts:
        # Grab current chunk
        if verbose:
            printnow("loading chunk starting at %d" % start)
        raw = l.get_chunk(start=start-prechunk, n_samples=bigchunksize)
        t = np.arange(start-prechunk, start - prechunk + bigchunksize) / float(fs) 

        # Now process one group at a time
        downsampled_l, t_l = [], []
        for group in channel_groups:
            # Column indexes into raw
            igroup = [l.header.Channel_ID.index(ch) for ch in group]
            
            # Mean and downsample
            #rawgroup = np.asarray([raw[ch] for ch in group])        
            meaned = np.mean(raw[:, igroup], axis=1)
            downsampled, new_t = scipy.signal.resample(meaned, post_bigchunksize, t=t, axis=0)
            
            # Slice out just desired chunk
            # If you grab one more sample here on the ends, you can check how
            # well the overlap is working between chunks
            downsampled = downsampled[post_prechunk:post_prechunk+post_chunksize]
            new_t = new_t[post_prechunk:post_prechunk+post_chunksize]
            
            # Append to result
            downsampled_l.append(downsampled)
            t_l.append(new_t)
        res_l.append(downsampled_l)
        rest_l.append(t_l)
    res = np.array(res_l)
    rest = np.array(rest_l)

    return res, rest



def _load_as_array6(l, channel_groups, starts, bigchunksize, prechunk, fs, 
    post_prechunk, post_chunksize, post_bigchunksize, verbose):
    """Avoid using intermediate dict object, mean first"""
    # Indexes into chunk columns
    ichannel_groups = [
        [l.header.Channel_ID.index(ch) for ch in group] 
        for group in channel_groups]
    
    # Set up return values
    res_l, rest_l = [], []
    for start in starts:
        # Grab current chunk
        if verbose:
            printnow("loading chunk starting at %d" % start)
        raw = l.get_chunk(start=start-prechunk, n_samples=bigchunksize)        
        t = np.arange(start-prechunk, start - prechunk + bigchunksize) / float(fs) 
        
        # Resample and slice out extra all at once
        dsraw, new_t = scipy.signal.resample(raw, post_bigchunksize, t=t, axis=0)
        dsraw = dsraw[post_prechunk:post_prechunk+post_chunksize]
        new_t = new_t[post_prechunk:post_prechunk+post_chunksize]

        # Now mean each group
        downsampled_l = [np.mean(dsraw[:, igroup], axis=1) 
            for igroup in ichannel_groups]
        t_l = [new_t] * len(ichannel_groups)
        res_l.append(downsampled_l)
        rest_l.append(t_l)    
    res = np.array(res_l)
    rest = np.array(rest_l)

    return res, rest