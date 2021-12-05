import json
import pprint
import re
import timeit
import copy
import math

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

    def utility(self, p):
        s = self._toString()
        val = 0
        if p == P1:
            for pattern, val1, _ in KILLS_P2:
                if re.findall(pattern, s):
                    return val1
            for pattern, val1, _ in KILLS_P1:
                if re.findall(pattern, s):
                    return val1
            for pattern, val1, _ in THREATS_P2:
                if re.findall(pattern, s):
                    return val1

            for pattern, val1, _ in THREATS_P1+VAL_P1+VAL_P2:
                val += val1*len(re.findall(pattern, s)) if not math.isnan(val1*len(re.findall(pattern, s))) else 0

        if p == P2:
            for pattern, _, val2 in KILLS_P1:
                if re.findall(pattern, s):
                    return val2
            for pattern, _, val2 in KILLS_P2:
                if re.findall(pattern, s):
                    return val2
            for pattern, _, val2 in THREATS_P1:
                if re.findall(pattern, s):
                    return val2

            for pattern, _, val2 in THREATS_P2+VAL_P1+VAL_P2:
                val += val2 * len(re.findall(pattern, s)) if not math.isnan(val2*len(re.findall(pattern, s))) else 0

        return val

    def toJson(self, filename):
        with open(filename, 'w') as f:
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

    def _toString(self):

        h = '\n'.join((''.join((str(_) for _ in line)) for line in self.board))
        v = '\n'.join((''.join((str(self.board[i][j]) for i in range(self.width))) for j in range(self.height)))

        d1 = '\n'.join((''.join((str(self.board[self.width - p + q - 1][q])
                                 for q in range(max(p - self.width + 1, 0), min(p + 1, self.height))))
                        for p in range(self.width + self.height - 1)))
        d2 = '\n'.join((''.join((str(self.board[p - q][q])
                                 for q in range(max(p - self.width + 1, 0), min(p + 1, self.height))))
                        for p in range(self.width + self.height - 1)))
        return h+'\n'+v+'\n'+d1+'\n'+d2+'\n'


class AIPlayer(object):

    MAX_DEPTH = 1

    def __init__(self):
        self.p1 = P1
        self.p2 = P2

    def terminalTest(self, depth):
        """终止条件测试

        Args:
            depth : 搜索树深度

        Returns:
            bool : 一个布尔值，表明是否达到截断搜索的深度
        """
        # TODO: CODE HERE

        if depth > self.MAX_DEPTH:
            return True

    def maxValue(self, board, alpha, beta, depth):
        """alpha-beta搜索的max节点

        Args:
            board : 当前棋盘状态
            alpha : alpha值
            beta : beta值
            depth : 当前搜索深度
        Returns:
            action: 落子位置（如D3）
            v: 这个action的效用值
        """
        # TODO: CODE HERE
        if self.terminalTest(depth):
            return board.utility(self.p2), (2, 2)

        val = float('-inf')

        action = None
        frontierLocal = copy.copy(board.frontier)

        cases = []

        for index in frontierLocal:

            boardCopy = Board(width=board.width, height=board.height)
            boardCopy.board = copy.deepcopy(board.board)
            boardCopy.frontier = copy.copy(board.frontier)
            boardCopy.addP1(index)

            # Local pruning
            valLocal = boardCopy.utility(P1)

            if valLocal == float('-inf'):
                continue

            elif valLocal == float('inf'):
                return valLocal, index

            cases.append((valLocal, boardCopy, index))

        if len(cases) == 1:
            return cases[0][0], cases[0][2]

        for _, boardCopy, index in sorted(cases, reverse=False, key=lambda x: x[0]):
            # print(_, index)
            moveVal, moveAction = self.minValue(boardCopy, alpha, beta, depth + 1)

            if moveVal >= val:
                val = moveVal
                action = index

            # if depth == 0:
            #     print(val, action)
                # boardCopy._display()

            if val >= beta:
                return val, action

            alpha = max(alpha, val)

        if not action:
            val = board.utility(self.p2)

        return val, action

    def minValue(self, board, alpha, beta, depth):
        """alpha-beta搜索的min节点

        Args:
            board : 当前棋盘状态
            alpha : alpha值
            beta : beta值
            depth : 当前搜索深度
        Returns:
            action: 落子位置（如D3）
            v: 这个action的效用值
        """
        # TODO: CODE HERE
        if self.terminalTest(depth):
            return board.utility(self.p1), None

        val = float('inf')

        action = None
        boardLocal = copy.copy(board.board)
        frontierLocal = copy.copy(board.frontier)

        cases = []

        for index in frontierLocal:

            boardCopy = Board(width=board.width, height=board.height)
            boardCopy.board = copy.deepcopy(board.board)
            boardCopy.frontier = copy.copy(board.frontier)
            boardCopy.addP2(index)

            # Local pruning
            valLocal = boardCopy.utility(P2)

            board.board = boardLocal
            board.frontier = frontierLocal

            if valLocal == float('inf'):
                continue

            elif valLocal == float('-inf'):
                return valLocal, index

            cases.append((valLocal, boardCopy, index))
        if len(cases) == 1:
            return cases[0][0], cases[0][2]

        for _, boardCopy, index in sorted(cases, reverse=True, key=lambda x: x[0]):
            moveVal, moveAction = self.maxValue(boardCopy, alpha, beta, depth + 1)

            if moveVal <= val:
                val = moveVal
                action = index

            if val <= alpha:
                return val, action

            beta = min(beta, val)

        if not action:
            val = board.utility(self.p2)

        return val, action

    def alphaBetaSearch(self, board):
        """[summary]

        Args:
            board ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """

        # TODO: CODE HERE

        depth = 0
        val, action = self.maxValue(board, float('-inf'), float('inf'), depth)
        return val, action

    def getMove(self, board):
        _, action = self.alphaBetaSearch(board)
        return action


if __name__ == "__main__":
    # t_util = timeit.timeit("from utils import Board;board = Board(20,20); "
    #                        "board.loadJson('board1.json');board.utility(1)",
    #                        number=1500)
    # t_numpy = timeit.timeit("from utils import Board; import numpy as np; import copy;"
    #                         " board = Board(20,20); "
    #                         " board.addP1((2,3));"
    #                         " x = np.array(board);"
    #                         " x.copy()",
    #                         number=1500)
    # t_tostring = timeit.timeit("from utils import Board; import copy;"
    #                            " board = Board(20,20); "
    #                            " board.addP1((2,3));"
    #                            " board._toString();",
    #                            number=1500)
    # print(t_util, t_numpy, t_tostring)
    # # # FIXME: Deepcopy is too expensive
    filename = 'C:/Users/Daniel/Desktop/Daniel/projects/FDU-Gomoku-Bot/code/Ver.beta/tmp/pbrain-pydan1.json'
    with open(filename, 'r') as f:
        b = json.load(f)
    board = Board(board=b)
    print(board)
    ai = AIPlayer()
    print(ai.getMove(board))
