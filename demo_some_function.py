import heapq
from solver import PosOfPlayer


def testLegalActions():
    allActions = [
        [-1, 0, "u", "U"],
        [1, 0, "d", "D"],
        [0, -1, "l", "L"],
        [0, 1, "r", "R"],
    ]
    posPlayer = tuple((2, 2))
    posBox = tuple((1, 2))

    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        print(x1, y1)
        if (x1, y1) in posBox:
            action.pop(2)
        else:
            action.pop(3)
        print(action[-1])


class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""

    def __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

    def print(self):
        print(self.Heap)


def test():
    """test action[[0]]"""
    actions = [[0]]
    a = actions.pop(0)
    arr = [1, 3, 4, 5, 5]


def testSlice():
    a = ["3", "34", "4", "34"]
    b = a[1:]
    print(b)


def testPriorityQueue():
    frontier = PriorityQueue()
    frontier.push((1, 2), 1)
    frontier.push((3, 2), 1)
    frontier.push((3, 10), 2)
    frontier.push((12, 5), 0)
    frontier.push((4, 3), 4)
    a = frontier.pop()
    print(a)
    frontier.print()


testSlice()
