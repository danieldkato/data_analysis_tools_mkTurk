import argparse
import logging
import os
import time
import concurrent.futures as cf

try:
    from .analyze_bystim import analyze_bystim
    from .allchan_scenefile import allchan_scenefile
    from .allchan_bl import allchan_bl
    from .allchan_wf import allchan_wf
    from .allchan_meanpsth import allchan_meanpsth
    from .allchan_objaverse import allchan_objaverse
    from .get_psth_objaverse import get_psth_objaverse
    from .get_wf_features import get_wf_features
except ImportError:
    from data_analysis_tools_mkTurk.analyze_bystim import analyze_bystim
    from data_analysis_tools_mkTurk.allchan_scenefile import allchan_scenefile
    from data_analysis_tools_mkTurk.allchan_bl import allchan_bl
    from data_analysis_tools_mkTurk.allchan_wf import allchan_wf
    from data_analysis_tools_mkTurk.allchan_meanpsth import allchan_meanpsth
    from data_analysis_tools_mkTurk.allchan_objaverse import allchan_objaverse
    from data_analysis_tools_mkTurk.get_psth_objaverse import get_psth_objaverse
    from data_analysis_tools_mkTurk.get_wf_features import get_wf_features

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
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting session processing for monkey={monkey}, date={date}, max_workers={max_workers}")
    total_start = time.time()
    
    # Stage 1: Parallel stimulus-based analysis (analyze_bystim.sh)
    logger.info(f"Stage 1/3: Starting analyze_bystim for {N_CHANNELS} channels...")
    stage1_start = time.time()
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ch in range(N_CHANNELS):
            futures.append(executor.submit(analyze_bystim, ch, monkey, date))
        for i, future in enumerate(cf.as_completed(futures)):
            future.result()
            if (i + 1) % 50 == 0:
                logger.info(f"  analyze_bystim progress: {i + 1}/{N_CHANNELS} channels completed")
    logger.info(f"Stage 1/3: analyze_bystim completed in {time.time() - stage1_start:.1f}s")
    
    # Stage 2: All-channel aggregation (template_all.sh)
    logger.info("Stage 2/3: Starting all-channel aggregation...")
    stage2_start = time.time()
    
    logger.info("  Running allchan_scenefile...")
    allchan_scenefile(monkey, date)
    
    logger.info("  Running allchan_bl...")
    allchan_bl(monkey, date)
    
    logger.info("  Running allchan_wf...")
    allchan_wf(monkey, date)
    
    logger.info("  Running allchan_meanpsth...")
    allchan_meanpsth(monkey, date)
    
    logger.info("  Running allchan_objaverse...")
    allchan_objaverse(monkey, date)
    
    logger.info(f"Stage 2/3: all-channel aggregation completed in {time.time() - stage2_start:.1f}s")
    
    # Stage 3: Parallel per-channel template processing (template_chan.sh)
    logger.info(f"Stage 3/3: Starting per-channel template processing for {N_CHANNELS} channels...")
    stage3_start = time.time()
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ch in range(N_CHANNELS):
            futures.append(executor.submit(_combined_template_chan, ch, monkey, date))
        for i, future in enumerate(cf.as_completed(futures)):
            future.result()
            if (i + 1) % 50 == 0:
                logger.info(f"  template_chan progress: {i + 1}/{N_CHANNELS} channels completed")
    logger.info(f"Stage 3/3: per-channel template processing completed in {time.time() - stage3_start:.1f}s")
    
    total_elapsed = time.time() - total_start
    logger.info(f"Processing for monkey {monkey} on date {date} completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a complete experimental session for neural recording data.')
    parser.add_argument('--monkey', type=str, required=True, help='Identifier for the monkey subject')
    parser.add_argument('--date', type=str, required=True, help='Date of the experimental session')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of worker processes (default: 8)')
    args = parser.parse_args()
    process_session_data(args.monkey, args.date, args.max_workers)