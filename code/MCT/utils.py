import json
import pprint
import re
import timeit
import copy
import math
import random

OUT = -1
BLANK = 0
P1 = 1
P2 = 2

KILLS_P1 = [
    (re.compile(r"11111"), float('inf'), float('inf'))
]
THREATS_P1 = [
    (re.compile(r"011110"), float('inf'), float('inf')),
    (re.compile(r"211110|011112"), 50, float('inf')),
    (re.compile(r"0011102|2011100|0011100"), 30, float('inf')),
    (re.compile(r"010110|011010"), 40, float('inf')),
]
VAL_P1 = [
    (re.compile(r"01110"), 8, 20),
    (re.compile(r'0110'), 2, 4)
]

KILLS_P2 = [
    (re.compile(r"22222"), float('-inf'), float('-inf'))
]
THREATS_P2 = [
    (re.compile(r"022220"), float('-inf'), float('-inf')),
    (re.compile(r"122220|022221"), float('-inf'), -50),
    (re.compile(r"0022201|1022200|0022200"), float('-inf'), -30),
    (re.compile(r"020220|022020"), float('-inf'), -40),
]
VAL_P2 = [
    (re.compile(r"02220"), -20, -8),
    (re.compile(r'0220'), -4, -2)
]


def kNN(x, y, k):
    yield x + k, y + k
    yield x + k, y - k
    yield x - k, y + k
    yield x - k, y - k
    yield x + k, y
    yield x - k, y
    yield x, y + k
    yield x, y - k


class Board(object):
    def __init__(self, width=0, height=0, k=1, board=None):  # k: kNN

        self.k = k
        self.frontier = set()
        if board:
            self.width = len(board[0])
            self.height = len(board)
            self.board = board

            # Make frontier
            for x in range(self.height):
                for y in range(self.width):
                    if self[x, y] > 0:
                        for i in range(1, self.k + 1):
                            for neighbour in kNN(x, y, i):
                                if self[neighbour] == BLANK:
                                    self.frontier.add(neighbour)

        else:
            self.width = width
            self.height = height
            assert self.width > 0 and self.height > 0, f'Width and height should be positive'
            self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]

    def __repr__(self):
        self._display()
        return f'Frontier: {self.frontier}'

    def __getitem__(self, index):
        """
            ALERT: Index by board[(x, y)] now !!!
            0: isFree
            1: isP1
            2: isP2
            -1: NOT VALID
        """
        if self.isValid(index):
            x, y = index
            return self.board[x][y]
        else:
            return -1

    def __setitem__(self, key, value):
        raise NotImplemented

    def getFrontier(self, k):
        frontier = set()
        for x in range(self.height):
            for y in range(self.width):
                if self[x, y] > 0:
                    for i in range(1, self.k + 1):
                        for neighbour in kNN(x, y, i):
                            if self[neighbour] == BLANK:
                                frontier.add(neighbour)
        return frontier

    def clear(self):
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.frontier = set()

    def isValid(self, index):
        x, y = index
        return 0 <= x < self.height and 0 <= y < self.width

    def addP1(self, index):
        """
        If the addition is temporary, we have to copy the previous frontier.
        If the addition is permanent, then do it without a copy.
        """
        assert self[index] == 0, f"Error occurs trying to play P1 at {index}\n" \
                                 f"{pprint.pformat(self.board)}"

        x, y = index
        self.board[x][y] = P1

        if (x, y) in self.frontier:
            self.frontier.remove((x, y))
        for i in range(1, self.k + 1):
            for neighbour in kNN(x, y, i):
                if self[neighbour] == BLANK:
                    self.frontier.add(neighbour)

    def addP2(self, index):
        """
        If the addition is temporary, we have to copy the previous frontier.
        If the addition is permanent, then do it without a copy.
        """
        assert self[index] == 0, f"Error occurs trying to play P1 at {index}\n" \
                                 f"{pprint.pformat(self.board)}"

        x, y = index
        self.board[x][y] = P2

        if (x, y) in self.frontier:
            self.frontier.remove((x, y))
        for i in range(1, self.k + 1):
            for neighbour in kNN(x, y, i):
                if self[neighbour] == BLANK:
                    self.frontier.add(neighbour)

    def remove(self, index):
        """
        Not a decent method!
        """
        x, y = index
        self.board[x][y] = 0

    def checkWin(self, lastMove, thisPlayer, monte=True, disp=False):
        x, y = lastMove
        h = ''.join(map(str, [self.board[x][i] for i in range(max(y-4, 0), min(y+5, self.height))]))
        v = ''.join(map(str, [self.board[i][y] for i in range(max(x-4, 0), min(x+5, self.height))]))

        d1 = ''.join(map(str, [self.board[x+i][y+i] for i in range(-min([x, y, 4]), min(self.width-max(x, y), 5))]))
        d2 = ''.join(map(str, [self.board[x+i][y-i] for i in range(-min([x, self.width-y-1, 4]), min([self.width-x-1, y, 5]))]))
        if disp:
            print(h, v, d1, d2)
        return self._checkWin(h, thisPlayer, monte) or self._checkWin(v, thisPlayer, monte) \
               or self._checkWin(d1, thisPlayer, monte) or self._checkWin(d2, thisPlayer, monte)

    def _checkWin(self, s, thisPlayer, monte=True):
        if monte:
            if thisPlayer == P1:
                return '11111' in s
            if thisPlayer == P2:
                return '22222' in s
        else:
            if thisPlayer == P1:
                return '11111' in s or '011110' in s
            if thisPlayer == P2:
                return '22222' in s or '022220' in s

    def toJson(self, filename):
        with open(filename, 'w+') as f:
            f.write(pprint.pformat(self.board))

    def loadJson(self, filename):
        with open(filename, 'r') as f:
            board = json.load(f)
            self.width = len(board[0])
            self.height = len(board)
            self.board = board

            # Make frontier
            for x in range(self.height):
                for y in range(self.width):
                    if self[x, y] > 0:
                        for i in range(1, self.k + 1):
                            for neighbour in kNN(x, y, i):
                                if self[neighbour] == BLANK:
                                    self.frontier.add(neighbour)

    def _display(self):
        # TODO: Visualize it
        print(f'   {" ".join(("%2d" %i for i in range(self.width)))}')
        display = [['-' for __ in range(self.width)] for _ in range(self.height)]
        for x in range(self.height):
            for y in range(self.width):
                if self[x, y] == 1:
                    display[x][y] = str(P1)
                elif self[x, y] == 2:
                    display[x][y] = str(P2)
                elif (x, y) in self.frontier:
                    display[x][y] = 'f'
        for i, row in enumerate(display):
            print(f'{"%-2d"%i} {"  ".join(row)}')

    def _checkPattern(self, line, thisPlayer):
        if thisPlayer == P1:
            f = re.finditer('01111', line)
        if thisPlayer == P2:
            pass


class AIPlayer(object):

    MAX_DEPTH = 1

    def __init__(self):
        self.p1 = P1
        self.p2 = P2

    def monteCarloSearch(self, board, move, max, intelligence=0.5):
        """[summary]

        Args:
            board ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """

        # TODO: CODE HERE
        added = []

        for _ in range(max):

            if random.random() < intelligence:
                frontierCopy = copy.copy(board.frontier)
                threatSpace = set()
                for k in range(1, 5):
                    threatSpace.update(frontierCopy.intersection(set(kNN(move[0], move[1], k))))

                # board._display()
                threatMoves = set()
                for index in threatSpace:
                    board.addP1(index)
                    if board.checkWin(index, P1, monte=False):
                        threatMoves.add(index)
                    board.remove(index)
                board.frontier = frontierCopy

                # board._display()
                if threatMoves:
                    frontier = threatMoves
                else:
                    frontier = copy.copy(board.frontier)
            else:
                frontier = copy.copy(board.frontier)

            move = random.choice(list(frontier))
            added.append(move)
            board.addP2(move)
            if board.checkWin(move, P2):
                for move in added:
                    board.remove(move)
                return -1

            if random.random() < intelligence:
                frontierCopy = copy.copy(board.frontier)
                threatSpace = set()
                for k in range(1, 5):
                    threatSpace.update(frontierCopy.intersection(set(kNN(move[0], move[1], k))))

                threatMoves = set()
                for index in threatSpace:
                    board.addP2(index)
                    if board.checkWin(index, P2, monte=False):
                        threatMoves.add(index)
                    board.remove(index)
                board.frontier = frontierCopy

                if threatMoves:
                    frontier = threatMoves
                    # print(frontier)
                else:
                    frontier = copy.copy(board.frontier)
            else:
                frontier = copy.copy(board.frontier)

            move = random.choice(list(frontier))
            added.append(move)
            board.addP1(move)
            if board.checkWin(move, P1):
                for move in added:
                    board.remove(move)
                return 1

        for move in added:
            board.remove(move)
        return 0

    def getMove(self, board, lastMove):
        frontier = board.getFrontier(2)

        threatMove = set()
        frontierCopy = copy.copy(board.frontier)
        for move in frontier:
            board.addP1(move)
            if board.checkWin(move, P1):
                board.remove(move)
                return move
            board.remove(move)

        for move in frontier:
            board.addP2(move)
            if board.checkWin(move, P2, monte=False):
                threatMove.add(move)
            board.remove(move)

        board.frontier = frontierCopy
        if threatMove:
            frontier.intersection_update(threatMove)

        frontier = {index: 0 for index in frontier}

        N = 0
        while N < 100:
            N = N + 1
            for index in frontier.keys():
                frontierCopy = copy.copy(board.frontier)
                board.addP1(index)
                frontier[index] += self.monteCarloSearch(board, index, max=10, intelligence=0)
                board.remove(index)
                board.frontier = frontierCopy

        boost_frontier = {max(frontier, key=frontier.get): 0*frontier.pop(max(frontier, key=frontier.get)) for i in range(len(frontier)//3+1)}

        N = 0
        while N < 100:
            N = N + 1
            for index in boost_frontier.keys():
                frontierCopy = copy.copy(board.frontier)
                board.addP1(index)
                boost_frontier[index] += self.monteCarloSearch(board, index, max=40, intelligence=0)
                board.remove(index)
                board.frontier = frontierCopy

        return max(boost_frontier, key=boost_frontier.get)


if __name__ == "__main__":
    pass
    # t_util = timeit.timeit("from utils import Board;board = Board(20,20); ",
    #                        number=1500)
    # t_deepcopy = timeit.timeit("from utils import Board; import copy;"
    #                            "board = Board(20,20); "
    #                            "board.addP1((2,3));"
    #                            "x = copy.deepcopy(board)",
    #                            number=15000)
    # t_mycopy = timeit.timeit("from utils import Board; import copy;"
    #                          "board = Board(20,20); "
    #                          "board.addP1((2,3));"
    #                          "boardCopy = [[c for c in line] for line in board.board]",
    #                          number=150000)
    # print(t_util, t_deepcopy, t_mycopy)
    # # # FIXME: Deepcopy is too expensive
    filename = 'tmp/board1.json'
    with open(filename, 'r') as f:
        b = json.load(f)
    board = Board(board=b, k=1)
    board._display()
    ai = AIPlayer()
    print(ai.getMove(board, None))
