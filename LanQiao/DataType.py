from collections import Counter
'''
列表
'''
number = list(['a', 'v', 'a', 'y'])
word_counts = Counter(number)
top_one = word_counts.most_common(1)
print(top_one)
'''
字典
'''
k =  [number[i] for i in range(3)]

dict1 = dict({
    'a':1,
    'b':2
})
del dict1['b']


'''
链表
'''
class Node:
    def __init__(self, data, pnext = None):
        self.data = data
        self._next = pnext
    def __repr__(self):
        return str(self.data)