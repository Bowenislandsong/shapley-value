
from typing import List, Any, Generator

def combinations(items: List[Any], r: int) -> Generator[List[Any], None, None]:
    if r == 0:
        yield []
    else:
        for i in range(len(items)):
            for combination in combinations(items[i+1:], r-1):
                yield [items[i]] + combination