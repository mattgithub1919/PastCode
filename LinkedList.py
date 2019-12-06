# Q1

# to better answer Q2, the LinkedQueue class includes both header and tail sentinel
class LinkedQueue:

    class _Node:

        def __init__(self, element, next):
            self._element = element
            self._next = next

    def __init__(self):
        self.head = None
        self.tail = None

    #self.head points to the most recently element in the queue
    #self.tail points to the first element in the queue
    def enqueue(self, val):
        temp = self._Node(val, self.head)
        self.head = temp
        # if LinkedQueue is empty, self.tail will need to point to the newly enqueued node
        # self.tail won't update if LinkedQueue is not empty
        if not self.tail:
            self.tail = temp

    def dequeue(self):
        temp = self.head
        result = self.tail
        try:
            # if there's only one element in LinkedList
            if temp._next == None:
                self.head = None
                self.tail = None
            while temp._next != self.tail:
                temp = temp._next
            self.tail = temp
            temp._next = None
        except AttributeError:
            print('LinkedQueue is empty!')
        return result._element

    def first(self):
        result = self.tail
        try:
            return result._element
        except AttributeError:
            print('LinkedQueue is empty!')

    def is_empty(self):
        return self.head == None

    def len(self):
        count = 0
        temp = self.head
        if self.is_empty:
            return 0
        while temp._next:
            count += 1
            temp = temp._next
        return count + 1

# Q2

# the following function is designed as a method under class LinkedQueue in Q1
def concatenate(self, Q2):
    if not Q2.is_empty():
        Q2.tail._next = self.head
        self.head = Q2.head
        if self.is_empty():
            self.tail = Q2.tail
        Q2.head = None
        Q2.tail = None

# Q3

# It is safe suppose that singly linked list L has an attribute of L.head
# suppose L.head points the node that recently inserted.
def reverse(L):
    curr = L.head
    try:
        after = curr.next
        before = None
        while after:
            curr.next = before
            before = curr
            curr = after
            after = after.next
        curr.next = before
        L.head = current
    except AttributeError:
        print('Given linked list is empty!')

# Q4

# In the design below, a SparseArray is initiated with A as parameter
class SparseArray:

    class Node:
        def __init__(self, index = None, value = None, next = None):
            self.idx = index
            self.val = value
            self.next = next
    
    # A singly linked list is built when initiating
    def __init__(self, A):
        self.head = None
        for i in range(len(A)):
            if A[i] != None:
                temp = self.Node(index = i, value = A[i], next = self.head)
                self.head = temp

    # need to first check if index j has already had a node
    # if yes, update value
    # if no, build a new node
    def __setitem__(self, j, e):
        curr = self.head
        try:
            while True:
                if curr.idx == j:
                    curr.val = e
                    break
                curr = curr.next
                if curr == None:
                    temp = self.Node(index = j, value = e, next = self.head)
                    self.head = temp
                    break
        except AttributeError:
            temp = self.Node(index = j, value = e, next = self.head)
            self.head = temp

    def __getitem__(self, j):
        temp = self.head
        try:
            while True:
                if temp.idx == j:
                    return temp.val
                temp = temp.next
                if temp == None:
                    return None
        except AttributeError:
            return None

""" 
Efficiency Analysis
1. self.__init__(A) - suppose there are n entries in A in total and m of them are non-empty. The constructor function will take O(n) running time and O(m) physical storage. Compared with O(n) running time and O(n) storage for sparse array A, it is physically more efficient.
2. self.__setitem__(j, e) - since it needs to browse all nodes, its time complexity is O(m). In a low level array, set item is O(1) due to contiguous physical storage. 
3. self.__getitem__(j) - it needs to browse all nodes too, thus time complexity is O(m). In a low level array, get item is O(1). 

In conclusion, SparseArray class is more physically efficient but less time efficient for standard indexing operations than low level array. 
 """
    
