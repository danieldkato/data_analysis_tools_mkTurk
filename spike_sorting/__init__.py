"""Spike-sorting pipelines (DARTsort, Kilosort4) and shared staging strategy.

Both pipelines preprocess a SpikeGLX recording and stage a single preprocessed
copy to fast disk before sorting, coordinated through one shared /local lock
(see staging.py) so a node never holds more than one staged recording at a time.
"""
