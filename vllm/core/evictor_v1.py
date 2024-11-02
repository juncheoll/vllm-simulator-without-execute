import enum
from abc import ABC, abstractmethod
from typing import OrderedDict
from sortedcontainers import SortedDict

import threading
import time


from vllm.block import PhysicalTokenBlock


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    LFU = enum.auto()
    LFUv2 = enum.auto()
    ARC = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> PhysicalTokenBlock:
        """Runs the eviction algorithm and returns the evicted block"""
        pass

    @abstractmethod
    def add(self, block: PhysicalTokenBlock):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        """Simply removes the block with the hash value block_hash from the
        evictor. Caller is responsible for making sure that block_hash is
        contained in the evictor before calling remove. Should be used to
        "bring back" blocks that have been freed but not evicted yet.
        """
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: OrderedDict[int, PhysicalTokenBlock] = OrderedDict()
        
        self.num_evict = 0
        self.num_add = 0
        self.num_remove = 0
        
        print('created LRUEvictor')
        self.thread = threading.Thread(target=self.log)
        #self.thread.start()
        
    def log(self):
        while True:
            time.sleep(10)
            print(f'LRU logging : evict : {self.num_evict}, '
                f'add : {self.num_add}, '
                f'remove : {self.num_remove}')

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.free_table

    def evict(self) -> PhysicalTokenBlock:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        self.num_evict += 1
        evicted_block = next(iter(self.free_table.values()))
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        for _, block in self.free_table.items():
            if evicted_block.last_accessed < block.last_accessed:
                break
            if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                evicted_block = block

        self.free_table.pop(evicted_block.block_hash)

        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.num_add += 1
        self.free_table[block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.num_remove += 1
        block: PhysicalTokenBlock = self.free_table[block_hash]
        self.free_table.pop(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class LFUEvictor(Evictor):

    def __init__(self):
        self.block_table = {}  # block_hash -> PhysicalTokenBlock
        self.freq_map = SortedDict()  # freq -> OrderedDict[block_hash, PhysicalTokenBlock]

        self.num_evict = 0
        self.num_add = 0
        self.num_remove = 0
        
        print('created LFUEvictor')
        self.thread = threading.Thread(target=self.log)
        #self.thread.start()
        
    def log(self):
        while True:
            time.sleep(10)
            print(f'LFU logging : evict : {self.num_evict}, '
                f'add : {self.num_add}, '
                f'remove : {self.num_remove}')
        
    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.block_table

    def evict(self) -> PhysicalTokenBlock:
        if not self.block_table:
            raise ValueError("No usable cache memory left")

        self.num_evict += 1
        
        min_freq = self.freq_map.peekitem(0)[0]
        # Evict the least frequently used block, using min_freq
        evicted_block_hash, evicted_block = self.freq_map[min_freq].popitem(last=False)

        if not self.freq_map[min_freq]:
            del self.freq_map[min_freq]

        del self.block_table[evicted_block_hash]
        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.num_add += 1
        
        self.block_table[block.block_hash] = block
        freq = block.freq
        if freq not in self.freq_map:
            self.freq_map[freq] = OrderedDict()
        self.freq_map[freq][block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.block_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        
        self.num_remove += 1
        
        block = self.block_table.pop(block_hash)
        freq = block.freq
        self.freq_map[freq].pop(block_hash)
        if not self.freq_map[freq]:
            del self.freq_map[freq]
        
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)
    
    
class LFUEvictorV2(Evictor):

    def __init__(self):
        self.block_table = {}  # block_hash -> PhysicalTokenBlock
        self.freq_map = SortedDict()  # freq -> OrderedDict[block_hash, PhysicalTokenBlock]
        self.num_evict = 0
        self.num_add = 0
        self.num_remove = 0
        
        print('created LFUEvictorV2')
        self.thread = threading.Thread(target=self.log)
        self.thread.start()
        
    def log(self):
        while True:
            time.sleep(10)
            print(f'LFU logging : evict : {self.num_evict}, '
                f'add : {self.num_add}, '
                f'remove : {self.num_remove}')
        
    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.block_table

    def _decay_frequency(self):
        #print('decay!!')
        decay_factor = 2  # 빈도를 절반으로 나눔
        to_move = []
        #print(self.freq_map.keys())
        
        freq_list = []
        for freq, block_dict in self.freq_map.items():
            if freq == 1 or freq in freq_list:
                continue
            freq_list.append(freq)
            #print(freq, len(block_dict))
            new_freq = freq // decay_factor  # 빈도가 1 아래로 떨어지지 않도록 조정
            if  new_freq not in self.freq_map:
                self.freq_map[new_freq] = OrderedDict()
            
            for block in block_dict.values():
                block.freq = new_freq
                to_move.append((freq, new_freq, block.block_hash))
        
        #print(to_move)
        for freq, new_freq, block_hash in to_move:    
            self.freq_map[new_freq][block_hash] = self.freq_map[freq].pop(block_hash)
        
                
        keys_to_delete = []
        for freq, block_dict in self.freq_map.items():
            if not block_dict:
                keys_to_delete.append(freq)
                    
        for freq in keys_to_delete:    
            del self.freq_map[freq]

    def evict(self) -> PhysicalTokenBlock:
        if not self.block_table:
            raise ValueError("No usable cache memory left")

        self.num_evict += 1
        
        if self.num_evict % 10000 == 0:
            self._decay_frequency()
        
        min_freq = self.freq_map.peekitem(0)[0]
        block_dict:OrderedDict[int, PhysicalTokenBlock] = self.freq_map[min_freq]
        # Evict the least frequently used block, using min_freq
        _, evicted_block = next(iter(block_dict.items()))
        
        for _, block in block_dict.items():
            if evicted_block.last_accessed < block.last_accessed:
                break
            if evicted_block.num_hashed_tokens < block.num_hashed_tokens:
                evicted_block = block

        self.freq_map[min_freq].pop(evicted_block.block_hash)
        if not self.freq_map[min_freq]:
            del self.freq_map[min_freq]
        
        del self.block_table[evicted_block.block_hash]
        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        self.num_add += 1
        
        self.block_table[block.block_hash] = block
        freq = block.freq
        if freq not in self.freq_map:
            self.freq_map[freq] = OrderedDict()
        self.freq_map[freq][block.block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.block_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        
        self.num_remove += 1
        
        block = self.block_table.pop(block_hash)
        freq = block.freq
        self.freq_map[freq].pop(block_hash)
        if not self.freq_map[freq]:
            del self.freq_map[freq]
        
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)
    
    
    
class ARCEvictor(Evictor):
    """
    Adaptive Replacement Cache (ARC) implementation.
    This evicts blocks using a combination of recent and frequent access patterns,
    dynamically balancing between LRU and LFU to improve cache hit ratios.
    """

    def __init__(self):
        self.block_table = {}  # block_hash -> PhysicalTokenBlock
        self.t1 = OrderedDict()  # LRU for recent blocks
        self.b1 = OrderedDict()  # Ghost entries for t1 evictions
        self.t2 = OrderedDict()  # LFU for frequent blocks
        self.b2 = OrderedDict()  # Ghost entries for t2 evictions
        self.target_t1_size = 0  # Target size for t1 (adaptively adjusted)

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.block_table

    def evict(self) -> PhysicalTokenBlock:
        if not self.block_table:
            raise ValueError("No usable cache memory left")

        if len(self.t1) > self.target_t1_size:
            # Evict from t1
            evicted_block_hash, evicted_block = self.t1.popitem(last=False)
            self.b1[evicted_block_hash] = evicted_block
        else:
            # Evict from t2
            evicted_block_hash, evicted_block = self.t2.popitem(last=False)
            self.b2[evicted_block_hash] = evicted_block

        del self.block_table[evicted_block_hash]
        evicted_block.computed = False
        return evicted_block

    def add(self, block: PhysicalTokenBlock):
        block_hash = block.block_hash

        if block_hash in self.t1:
            # Promote to t2
            self.t1.pop(block_hash)
            self.t2[block_hash] = block
        elif block_hash in self.b1:
            # Increase target_t1_size since a block from b1 is re-added
            self.target_t1_size = min(self.target_t1_size + 1, len(self.block_table))
            self.b1.pop(block_hash)
            self.t2[block_hash] = block
        elif block_hash in self.t2:
            # Update t2
            self.t2.move_to_end(block_hash)
        elif block_hash in self.b2:
            # Decrease target_t1_size since a block from b2 is re-added
            self.target_t1_size = max(self.target_t1_size - 1, 0)
            self.b2.pop(block_hash)
            self.t2[block_hash] = block
        else:
            # Add to t1
            if len(self.block_table) >= len(self.block_table):
                self.evict()
            self.t1[block_hash] = block

        self.block_table[block_hash] = block

    def remove(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.block_table:
            raise ValueError("Attempting to remove block that's not in the evictor")

        block = self.block_table.pop(block_hash)
        if block_hash in self.t1:
            self.t1.pop(block_hash)
        elif block_hash in self.t2:
            self.t2.pop(block_hash)
        elif block_hash in self.b1:
            self.b1.pop(block_hash)
        elif block_hash in self.b2:
            self.b2.pop(block_hash)

        return block

    def get(self, block_hash: int) -> PhysicalTokenBlock:
        if block_hash not in self.block_table:
            raise ValueError("Block not found in the evictor")

        block = self.block_table[block_hash]
        if block_hash in self.t1:
            # Promote to t2
            self.t1.pop(block_hash)
            self.t2[block_hash] = block
        elif block_hash in self.t2:
            # Update t2
            self.t2.move_to_end(block_hash)
        return block

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)
    

def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.LFU:
        return LFUEvictor()
    elif eviction_policy == EvictionPolicy.LFUv2:
        return LFUEvictorV2()
    elif eviction_policy == EvictionPolicy.ARC:
        return ARCEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
