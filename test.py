import heapq

heap = []
heapq.heappush(heap, (2, 'two')), print(heap)
heapq.heappush(heap, (3, 'three')), print(heap)
heapq.heappush(heap, (1, 'one')), print(heap)
heapq.heappush(heap, (2, 'two-2')), print(heap)

print(heapq.heappop(heap)) , print(heap)
print(heapq.heappop(heap))
print(heapq.heappop(heap))