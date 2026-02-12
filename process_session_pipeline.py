import os
import concurrent.futures as cf

from .analyze_bystim import analyze_bystim

from .allchan_scenefile import allchan_scenefile
from .allchan_bl import allchan_bl
from .allchan_wf import allchan_wf
from .allchan_meanpsth import allchan_meanpsth
from .allchan_objaverse import allchan_objaverse

from .get_psth_objaverse import get_psth_objaverse
from .get_wf_features import get_wf_features

def _combined_template_chan(chan: int, monkey: str, date: str):
    """Process channel glued function: compute PSTH from Objaverse data and extract waveform features."""
    get_psth_objaverse(chan, monkey, date)
    get_wf_features(chan, monkey, date)

def process_session_data(monkey: str, date: str, max_workers: int | None = 8):
    """
    Process a complete experimental session for a given monkey and date.
    This function orchestrates the complete analysis pipeline for neural recording data,
    including stimulus-based analysis, channel aggregation, and individual channel processing.

    Args:
        monkey (str): Identifier for the monkey subject.
        date (str): Date of the experimental session in string format.
        max_workers (int | None, optional): Maximum number of worker processes for parallel 
            execution. Defaults to 8. If None, uses the default ProcessPoolExecutor behavior.

    Returns:
        None: This function performs file I/O operations and prints completion status.

    Notes:
        - Processes data from 384 channels (N_CHANNELS constant).
        - The pipeline consists of three main stages:
            1. Parallel stimulus-based analysis across all channels (analyze_bystim)
            2. All-channel aggregation operations:
               - Scene file generation (allchan_scenefile)
               - Baseline processing (allchan_bl)
               - Waveform analysis (allchan_wf)
               - Mean PSTH calculation (allchan_meanpsth)
               - Object aversion analysis (allchan_objaverse)
            3. Parallel per-channel template processing (_combined_template_chan)
        - Uses ProcessPoolExecutor for parallel processing to improve performance.
        - All submitted futures are awaited for completion before proceeding.

    Example:
        >>> process_session("West", "20240115", max_workers=8)
        Processing for monkey West on date 2024-01-15 completed.
    """
    N_CHANNELS = 384
    
    # Original analyze_bystim.sh
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ch in range(N_CHANNELS):
            futures.append(executor.submit(analyze_bystim, ch, monkey, date))
        for future in cf.as_completed(futures):
            future.result()
    
    
    # Original template_all.sh
    allchan_scenefile(monkey, date)
    allchan_bl(monkey, date)
    allchan_wf(monkey, date)
    allchan_meanpsth(monkey, date)
    allchan_objaverse(monkey, date)
    
    
    # Original templace_chan.sh
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ch in range(N_CHANNELS):
            futures.append(executor.submit(_combined_template_chan, ch, monkey, date))
        for future in cf.as_completed(futures):
            future.result()
    
    
    print(f"Processing for monkey {monkey} on date {date} completed.")