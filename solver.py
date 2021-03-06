import sys
import collections
import numpy as np
import heapq
import time
import numpy as np

global posWalls, posGoals


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


"""Load puzzles and define the rules of sokoban"""


def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace("\n", "") for x in layout]
    layout = [",".join(layout[i]) for i in range(len(layout))]
    layout = [x.split(",") for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == " ":
                layout[irow][icol] = 0  # free space
            elif layout[irow][icol] == "#":
                layout[irow][icol] = 1  # wall
            elif layout[irow][icol] == "&":
                layout[irow][icol] = 2  # player
            elif layout[irow][icol] == "B":
                layout[irow][icol] = 3  # box
            elif layout[irow][icol] == ".":
                layout[irow][icol] = 4  # goal
            elif layout[irow][icol] == "X":
                layout[irow][icol] = 5  # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum - colsNum)])

    # print(layout)
    return np.array(layout)


def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp


def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0])  # e.g. (2, 2)


def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(
        tuple(x)
        for x in np.argwhere(
            (gameState == 3) | (gameState == 5)
        )  # argwhere : tr??? v??? t???a ????? v??? tr?? th???a ??k trong ngo???c
    )  # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))


def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1))  # e.g. like those above


def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(
        tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))
    )  # e.g. like those above


def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)


def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper():  # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls


def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [
        [-1, 0, "u", "U"],
        [1, 0, "d", "D"],
        [0, -1, "l", "L"],
        [0, 1, "r", "R"],
    ]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox:  # the move was a push
            action.pop(2)  # drop the little letter
        else:
            action.pop(3)  # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else:
            continue
    return tuple(tuple(x) for x in legalActions)  # e.g. ((0, -1, 'l'), (0, 1, 'R'))


def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer  # the previous position of player
    newPosPlayer = [
        xPlayer + action[0],
        yPlayer + action[1],
    ]  # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper():  # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox


def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1],
        [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1],
    ]
    flipPattern = [
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        [2, 1, 0, 5, 4, 3, 8, 7, 6][::-1],
        [0, 3, 6, 1, 4, 7, 2, 5, 8][::-1],
    ]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [
                (box[0] - 1, box[1] - 1),
                (box[0] - 1, box[1]),
                (box[0] - 1, box[1] + 1),
                (box[0], box[1] - 1),
                (box[0], box[1]),
                (box[0], box[1] + 1),
                (box[0] + 1, box[1] - 1),
                (box[0] + 1, box[1]),
                (box[0] + 1, box[1] + 1),
            ]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posWalls
                    and newBoard[5] in posWalls
                ):
                    return True
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posWalls
                    and newBoard[5] in posBox
                ):
                    return True
                elif (
                    newBoard[1] in posBox
                    and newBoard[2] in posBox
                    and newBoard[5] in posBox
                ):
                    return True
                elif (
                    newBoard[1] in posBox
                    and newBoard[6] in posBox
                    and newBoard[2] in posWalls
                    and newBoard[3] in posWalls
                    and newBoard[8] in posWalls
                ):
                    return True
    return False


"""Implement all approcahes"""


def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)  # ((2,2),((1,2),(3,2),(4,2)))
    print(startState)

    # t???o m???t h??ng ?????i frontier

    frontier = collections.deque([[startState]])

    exploredSet = set()  # t???p c??c state ???? gh??
    actions = [[0]]
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):  # check current Pos Box
            temp += node_action[1:]  # c?? ph??p get row th??? 1 c???a matrix
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp


def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    # l???y v??? tr?? Box v?? Player ban ?????u t??? map - gameState
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    # kh???i t???o state ban ?????u
    startState = (
        beginPlayer,
        beginBox,
    )

    frontier = collections.deque([[startState]])  # t???o queue l??u tr???ng th??i c??c state
    actions = collections.deque([[0]])  # t???o queue l??u d??y c??c action t????ng ???ng
    exploredSet = set()  # t???o Set l??u c??c node ???? ??i qua
    temp = []
    ### Implement breadthFirstSearch here
    while frontier:
        node = frontier.popleft()  # l???y node t??? queue
        node_action = actions.popleft()  # l???y d??y action t????ng ???ng t??? queue
        if isEndState(node[-1][-1]):  # ki???m tra node n??y c?? ph???i Goal ko
            temp += node_action[
                1:
            ]  # n???u l?? Goal th?? tr??? v??? d??y action t????ng ???ng -> k???t qu??? b??i to??n
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])  # th??m node v??o t???p node ???? ??i qua

            # duy???t qua c??c action h???p l??? c???a Player (l??n, xu???ng, tr??i ph???i, ?????y, ko ?????y)
            # h??m legalActions tr??? v??? c??c action c???a Player c?? th??? th???c hi???n khi ??? state hi???n t???i
            for action in legalActions(node[-1][0], node[-1][1]):
                # t???o state m???i t??? state hi???n t???i v?? action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # ki???m tra th??ng c?? b??? ?????y v??o g??c t?????ng kh??ng
                if isFailed(newPosBox):
                    continue

                # th??m node ???? v??o queue
                frontier.append(node + [(newPosPlayer, newPosBox)])

                # th??m d??y c??c actions t????ng ???ng v??o queue
                actions.append(node_action + [action[-1]])
    return temp


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    # l???y v??? tr?? ban ?????u c???a Player v?? v??? tr?? c??c H???p
    beginBox = PosOfBoxes(gameState)  # Ex: (2,3)
    beginPlayer = PosOfPlayer(gameState)  # Ex: ((1,2),(2,3),(3,4))

    startState = (beginPlayer, beginBox)  # tuple l??u tr???ng th??i ban ?????u
    frontier = PriorityQueue()  # t???o m???t h??ng ?????i ??u ti??n
    frontier.push([startState], 0)  # ????? v??? tr?? ban ?????u v??o h??ng ?????i v???i chi ph?? l?? 0
    exploredSet = set()  # T???o m???t t???p h???p ????? ki???m tra c??c node ???? ??i r???i -> ko ??i l???i
    actions = (
        PriorityQueue()
    )  # h??ng ?????i ch???a d??yc??c action ????? ??i ?????n 1 tr???ng th??i t????ng ???ng trong frontier
    actions.push([0], 0)
    temp = []
    ### Implement uniform cost search here

    while frontier.isEmpty() == False:
        node = frontier.pop()  # l???y trong h??ng ?????i node c?? chi ph?? nh??? nh???t ????? ??i
        node_action = actions.pop()  # l???y d??y action t????ng ???ng v???i node
        if isEndState(node[-1][-1]):  # ki???m tra c?? ph???i l?? goal kh??ng
            temp += node_action[
                1:
            ]  # n???u l?? goal th?? tr??? v??? d??y action t????ng ???ng -> k???t qu??? b??i to??n
            break
        if node[-1] not in exploredSet:  # n???u node n??y ch??a th??m
            exploredSet.add(node[-1])  # th??m node v??o t???p c??c node ???? th??m

            # v??ng l???p duy???t qua c??c action h???p l???
            # legalActions: tr??? v??? d??y c??c action h???p l??? c???a tr???ng th??i (l??n, xu???ng, tr??i, ph???i, ?????y th??ng hay kh??ng, ....)
            for action in legalActions(node[-1][0], node[-1][1]):
                # c???p nh???t state m???i d???a v??o state hi???n t???i v?? action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # h??m isFailed ki???m tra v??? tr?? th??ng c?? b??? ?????y v??o g??c t?????ng kh??ng
                if isFailed(newPosBox):
                    continue

                # c???p nh???t state m???i v??o d??y c??c state tr?????c ????
                newListStates = node + [(newPosPlayer, newPosBox)]

                # c???p nh???t action m???i v?? d??y actions
                newListActions = node_action + [action[-1]]

                # t??nh chi ph?? c???a d??y actions m???i
                stateCost = cost(newListActions[1:])

                # th??m node n??y v??o h??ng ?????i ??u ti??n k??m tr???ng s???
                frontier.push(newListStates, stateCost)

                # th??m node v??o h??ng ?????i ??u ti??n k??m tr???ng s???
                actions.push(newListActions, stateCost)
    return temp


"""Read command"""


def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option(
        "-l",
        "--level",
        dest="sokobanLevels",
        help="level of game to play",
        default="level1.txt",
    )
    parser.add_option(
        "-m", "--method", dest="agentMethod", help="research method", default="bfs"
    )
    args = dict()
    options, _ = parser.parse_args(argv)
    with open("assets/levels/" + options.sokobanLevels, "r") as f:
        layout = f.readlines()
    args["layout"] = layout
    args["method"] = options.agentMethod
    return args


def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == "dfs":
        result = depthFirstSearch(gameState)
    elif method == "bfs":
        result = breadthFirstSearch(gameState)
    elif method == "ucs":
        result = uniformCostSearch(gameState)
    else:
        raise ValueError("Invalid method.")
    time_end = time.time()
    print("Runtime of %s: %.2f second." % (method, time_end - time_start))
    print(result)
    return result
