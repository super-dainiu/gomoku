import json
import pprint
import re
import time
import copy
import math

OUT = -1
BLANK = 0
P1 = 1
P2 = 2

PENALTY = 1.5
DISCOUNT1 = 1
DISCOUNT2 = 0.9
DISCOUNT3 = 0.1

SCORE = {
    # Winning Threats
    "FIVE": 100000,                 # ooooo
    "OPEN_FOUR": 10000,             # _oooo_
    # Forcing Threats
    "SIMPLE_FOUR": 1000,            # xoooo_ / oo_oo / ooo_o
    "OPEN_THREE": 1000,             # __ooo__
    "SIMPLE_THREE": 900,            # _oo_o_ / x_ooo__
    "COMPENSATION": 630,            # o_o_o_o_o -> 3 * o_o_o + 630 = 1000
    # Potential Threats
    "OPEN_TWO": 100,                # __oo__ / __o_o__
    "BROKEN_THREE": 90,             # xooo__ / xoo_o_ / xo_oo_ / x_ooo_x / o_o_o
    "SIMPLE_TWO": 90,               # x_oo__ / x_o_o__
}

# THREATS_P1 = [
#     (re.compile(r'11111'), SCORE['FIVE'], SCORE['FIVE']),
#     (re.compile(r'011110'), SCORE['OPEN_FOUR'], SCORE['OPEN_FOUR']),
#     (re.compile(r'211110|011112|11011|11101|10111'), SCORE['SIMPLE_FOUR'], SCORE['SIMPLE_FOUR']),
#     (re.compile(r'0011100'), SCORE['OPEN_THREE'], SCORE['OPEN_THREE']),
#     (re.compile(r'011010|010110|2011100|0011102'), SCORE['SIMPLE_THREE'], SCORE['SIMPLE_THREE']),
#     (re.compile(r'101010101'), SCORE['COMPENSATION'], SCORE['COMPENSATION']),
#     (re.compile(r'001100|0010100'), SCORE['OPEN_TWO'], SCORE['OPEN_TWO']),
#     (re.compile(r'211100|001112|211010|010112|210110|011012|2011102|10101'), SCORE['BROKEN_THREE'], SCORE['BROKEN_THREE']),
#     (re.compile(r'201100|001102|2010100|0010102'), SCORE['SIMPLE_TWO'], SCORE['SIMPLE_TWO'])
# ]
#
# THREATS_P2 = [
#     (re.compile(r'22222'), SCORE['FIVE'], SCORE['FIVE']),
#     (re.compile(r'022220'), SCORE['OPEN_FOUR'], SCORE['OPEN_FOUR']),
#     (re.compile(r'122220|022221|22022|22202|20222'), SCORE['SIMPLE_FOUR'], SCORE['SIMPLE_FOUR']),
#     (re.compile(r'0022200'), SCORE['OPEN_THREE'], SCORE['OPEN_THREE']),
#     (re.compile(r'022020|020220|1022200|0022201'), SCORE['SIMPLE_THREE'], SCORE['SIMPLE_THREE']),
#     (re.compile(r'202020202'), SCORE['COMPENSATION'], SCORE['COMPENSATION']),
#     (re.compile(r'002200|0020200'), SCORE['OPEN_TWO'], SCORE['OPEN_TWO']),
#     (re.compile(r'122200|002221|122020|020221|120220|022021|1022201|20202'), SCORE['BROKEN_THREE'], SCORE['BROKEN_THREE']),
#     (re.compile(r'102200|002201|1020200|0020201'), SCORE['SIMPLE_TWO'], SCORE['SIMPLE_TWO']),
# ]

THREATS_P1 = [
    (re.compile(r"11111"), 1000000, 1000000*DISCOUNT3),
    (re.compile(r"011110"), 100000, 100000*DISCOUNT3),
    (re.compile(r"211110|011112|11011|11101|10111"), 1000, 1000*DISCOUNT3),
    (re.compile(r"0011100|010110|011010"), 1000, 1000*DISCOUNT3),
    (re.compile(r"2011100|0011102"), 1000*DISCOUNT1, 1000*DISCOUNT1*DISCOUNT3),
    (re.compile(r"001100"), 100, 100*DISCOUNT3),
    (re.compile(r"211010|010112|210110|011012|0010100|211100|001112|10101"), 100*DISCOUNT2, 100*DISCOUNT2*DISCOUNT3),
    (re.compile(r"201100|001102|2011102"), 100*DISCOUNT2*DISCOUNT2, 100*DISCOUNT2*DISCOUNT2*DISCOUNT3),
    (re.compile(r"2010100|0010102"), 10, 10*DISCOUNT3),
    (re.compile(r"2010102"), 10*DISCOUNT2, 10*DISCOUNT2*DISCOUNT3)
]

THREATS_P2 = [
    (re.compile(r"22222"), -1000000*PENALTY*DISCOUNT3, -1000000*PENALTY),
    (re.compile(r"022220"), -100000*PENALTY*DISCOUNT3, -100000*PENALTY),
    (re.compile(r"122220|022221|22022|22202|20222"), -1000*PENALTY*DISCOUNT3, -1000*PENALTY),
    (re.compile(r"020220|022020|0022200"), -1000*PENALTY*DISCOUNT3, -1000*PENALTY),
    (re.compile(r"1022200|0022201"), -1000*DISCOUNT1*PENALTY*DISCOUNT3, -1000*DISCOUNT1*PENALTY),
    (re.compile(r"002200"), -100*DISCOUNT3, -100),
    (re.compile(r"122020|020221|120220|022021|0020200|122200|002221|20202"), -100*DISCOUNT2*DISCOUNT3, -100*DISCOUNT2),
    (re.compile(r"102200|002201|1022201"), -100*DISCOUNT2*DISCOUNT2*DISCOUNT3, -100*DISCOUNT2*DISCOUNT2),
    (re.compile(r"1020200|0020201"), -10*DISCOUNT3, -10),
    (re.compile(r"1020201"), -10*DISCOUNT2*DISCOUNT3, -10*DISCOUNT2)
]


# THREATS_P1 = [
#     (re.compile(r"11111"), 1000000, 1000000),
#     (re.compile(r"011110"), 10000, 10000),
#     (re.compile(r"211110|011112|11011|11101|10111"), 1000, 1000*DISCOUNT3),
#     (re.compile(r"0011100"), 1000, 1000*DISCOUNT3),
#     (re.compile(r"010110|011010|2011100|0011102"), 1000*DISCOUNT1, 1000*DISCOUNT1*DISCOUNT3),
#     (re.compile(r"0010100|211100|001112|10101"), 100*DISCOUNT2, 100*DISCOUNT2*DISCOUNT3),
# ]
#
# THREATS_P2 = [
#     (re.compile(r"22222"), -1000000*PENALTY, -1000000*PENALTY),
#     (re.compile(r"022220"), -10000*PENALTY, -10000*PENALTY),
#     (re.compile(r"122220|022221|22022|22202|20222"), -1000*DISCOUNT3*PENALTY, -1000*PENALTY),
#     (re.compile(r"0022200"), -1000*DISCOUNT3*PENALTY, -1000*PENALTY),
#     (re.compile(r"020220|022020|1022200|0022201"), -1000*DISCOUNT1*DISCOUNT3*PENALTY, -1000*DISCOUNT1*PENALTY),
#     (re.compile(r"002200"), -100, -100*DISCOUNT3),
#     (re.compile(r"0020200|122200|002221|20202"), -100*DISCOUNT2*DISCOUNT3, -100*DISCOUNT2),
#     (re.compile(r"122020|020221|120220|022021|102200|002201|1022201"), -100*DISCOUNT2*DISCOUNT2*DISCOUNT3, -100*DISCOUNT2*DISCOUNT2),
# ]


# THREATS_P1 = [
#     (re.compile(r"11111"), 1000000, 1000000, None, None, 0),
#     (re.compile(r"011110"), 10000, 10000, (0, 5), (0, 5), '1killing'),
#     (re.compile(r"211110"), 1000, 1000*DISCOUNT3, 5, 5, '1killing'),
#     (re.compile(r"011112"), 1000, 1000*DISCOUNT3, 0, 0, '1killing'),
#     (re.compile(r"11011"), 1000, 1000*DISCOUNT3, 2, 2, '1killing'),
#     (re.compile(r"11101"), 1000, 1000*DISCOUNT3, 3, 3, '1killing'),
#     (re.compile(r"10111"), 1000, 1000*DISCOUNT3, 1, 1, '1killing'),
#     (re.compile(r"0011100"), 1000, 1000*DISCOUNT3, 1, 5, '1challenging'),
#     (re.compile(r"010110"), 1000*DISCOUNT1, 1000 * DISCOUNT1 * DISCOUNT3, 2, (0, 2, 5), '1challenging'),
#     (re.compile(r"011010"), 1000 * DISCOUNT1, 1000 * DISCOUNT1 * DISCOUNT3, 3, (0, 3, 5), '1challenging'),
#     (re.compile(r"2011100"), 1000 * DISCOUNT1, 1000 * DISCOUNT1 * DISCOUNT3, 5, (1, 5, 6), '1challenging'),
#     (re.compile(r"0011102"), 1000 * DISCOUNT1, 1000 * DISCOUNT1 * DISCOUNT3, 1, (0, 1, 5), '1challenging'),
#     (re.compile(r"001100"), 100, 100*DISCOUNT3, None, None, 0),
#     (re.compile(r"0010100|211100|001112|10101"), 100*DISCOUNT2, 100*DISCOUNT2*DISCOUNT3, None, None, 0),
#     (re.compile(r"211010|010112|210110|011012|201100|001102|2011102"), 100*DISCOUNT2*DISCOUNT2, 100*DISCOUNT2*DISCOUNT2*DISCOUNT3, None, None, 0),
#     (re.compile(r"2010100|0010102"), 10, 10*DISCOUNT3, None, None, 0),
#     (re.compile(r"2010102"), 10*DISCOUNT2, 10*DISCOUNT2*DISCOUNT3, None, None, 0)
# ]
#
# THREATS_P2 = [
#     (re.compile(r"22222"), -1000000*PENALTY, -1000000*PENALTY, None, None, 0),
#     (re.compile(r"022220"), -10000*PENALTY, -10000*PENALTY, (0, 5), (0, 5), '2killing'),
#     (re.compile(r"122220"), -1000*DISCOUNT3*PENALTY, -1000*PENALTY, 5, 5, '2killing'),
#     (re.compile(r"022221"), -1000 * DISCOUNT3 * PENALTY, -1000 * PENALTY, 0, 0, '2killing'),
#     (re.compile(r"22022"), -1000 * DISCOUNT3 * PENALTY, -1000 * PENALTY, 2, 2, '2killing'),
#     (re.compile(r"22202"), -1000 * DISCOUNT3 * PENALTY, -1000 * PENALTY, 3, 3, '2killing'),
#     (re.compile(r"20222"), -1000 * DISCOUNT3 * PENALTY, -1000 * PENALTY, 1, 1, '2killing'),
#     (re.compile(r"0022200"), -1000*DISCOUNT3*PENALTY, -1000*PENALTY, 1, 5, '2challenging'),
#     (re.compile(r"020220"), -1000*DISCOUNT1*DISCOUNT3*PENALTY, -1000*DISCOUNT1*PENALTY, 2, (0, 2, 5), '2challenging'),
#     (re.compile(r"022020"), -1000 * DISCOUNT1 * DISCOUNT3 * PENALTY, -1000 * DISCOUNT1 * PENALTY, 3, (0, 3, 5), '2challenging'),
#     (re.compile(r"1022200"), -1000 * DISCOUNT1 * DISCOUNT3 * PENALTY, -1000 * DISCOUNT1 * PENALTY, 5, (1, 5, 6), '2challenging'),
#     (re.compile(r"0022201"), -1000 * DISCOUNT1 * DISCOUNT3 * PENALTY, -1000 * DISCOUNT1 * PENALTY, 1, (0, 1, 5), '2challenging'),
#     (re.compile(r"002200"), -100, -100*DISCOUNT3, None, None, 0),
#     (re.compile(r"0020200|122200|002221|20202"), -100*DISCOUNT2*DISCOUNT3, -100*DISCOUNT2, None, None, 0),
#     (re.compile(r"122020|020221|120220|022021|102200|002201|1022201"), -100*DISCOUNT2*DISCOUNT2*DISCOUNT3, -100*DISCOUNT2*DISCOUNT2, None, None, 0),
#     (re.compile(r"1020200|0020201"), -10*DISCOUNT3, -10, None, None, 0),
#     (re.compile(r"1020201"), -10*DISCOUNT2*DISCOUNT3, -10*DISCOUNT2, None, None, 0)
# ]

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
    def __init__(self, width=0, height=0, k=2, board=None):  # k: kNN

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

        self.value = self._utility()
        self.threat = {'1killing': None, '1challenging': None, '2killing': None, '2challenging': None}

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

    def addP2(self, index):
        """
        If the addition is temporary, we have to copy the previous frontier.
        If the addition is permanent, then do it without a copy.
        """
        assert self[index] == 0, f"Error occurs trying to play P2 at {index}\n" \
                                 f"{pprint.pformat(self.board)}"

        x, y = index
        self.board[x][y] = P2


    def remove(self, index):
        """
        Not a decent method!
        """
        x, y = index
        self.board[x][y] = 0

    def localUtility(self, lastMove):
        x, y = lastMove
        h = ''.join(map(str, [self.board[x][i] for i in range(0, self.height)]))
        v = ''.join(map(str, [self.board[i][y] for i in range(0, self.height)]))

        d1 = ''.join(
            map(str, [self.board[x + i][y + i] for i in range(-min([x, y]), self.width - max(x, y))]))
        d2 = ''.join(map(str, [self.board[x + i][y - i] for i in
                               range(-min([x, self.width - y - 1]), min([self.width - x, y + 1]))]))

        s = h+'\n'+v+'\n'+d1+'\n'+d2+'\n'

        valP1, valP2 = 0, 0
        for pattern, val1, val2 in THREATS_P1 + THREATS_P2:
            valP1 += val1 * len(re.findall(pattern, s))
            valP2 += val2 * len(re.findall(pattern, s))
        return valP1, valP2

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

    def _updateFrontier(self, index):
        x, y = index
        if (x, y) in self.frontier:
            self.frontier.remove((x, y))
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

    def _utility(self):
        s = self._toString()
        valP1, valP2 = 0, 0
        for pattern, val1, val2 in THREATS_P1 + THREATS_P2:
            valP1 += val1 * len(re.findall(pattern, s))
            valP2 += val2 * len(re.findall(pattern, s))

        return valP1, valP2

    # def _threatSpace(self):
    #     threat = {'1killing': set(), '1challenging': set(), '2killing': set(), '2challenging': set()}
    #     h = '\n'.join((''.join((str(_) for _ in line)) for line in self.board))
    #     v = '\n'.join((''.join((str(self.board[i][j]) for i in range(self.width))) for j in range(self.height)))
    #     for pattern, _, _, _, _, threatType in THREATS_P1 + THREATS_P2:

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

    MAX_DEPTH = 4

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

    def maxValue(self, board, alpha, beta, depth, timeStart=None):
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
            return board.value[0], None

        val = float('-inf')

        action = None
        potential = None
        frontierLocal = tuple(board.frontier)
        valLocal = board.value

        argList = []
        # calculate the current value
        for index in frontierLocal:
            valP1, valP2 = valLocal
            dvalP1, dvalP2 = board.localUtility(index)
            valP1 -= dvalP1
            valP2 -= dvalP2

            board.addP1(index)
            dvalP1, dvalP2 = board.localUtility(index)
            if depth == 0 and dvalP1 > 500000:
                action = index
            if depth == 0 and dvalP1 > 50000:
                potential = index

            if abs(valP1) > 500000 or abs(valP2) > 500000:
                dvalP1 *= 0
                dvalP2 *= 0

            valP1 += dvalP1
            valP2 += dvalP2
            board.remove(index)

            if depth == 0:
                board.addP2(index)
                dvalP1, dvalP2 = board.localUtility(index)
                if dvalP1 < -3000*PENALTY:
                    valP1 += 1000
                    valP2 += 1000
                if depth == 0 and dvalP1 < -300000 and not action:
                    action = index
                board.remove(index)

            value = (valP1, valP2)

            argList.append((index, value))

        if depth + 1 > self.MAX_DEPTH:
            sum = 0
            for index, value in argList:
                sum += value[1]
            return sum/len(argList), action

        argList.sort(key=lambda x: x[1][1], reverse=True)
        argList = argList[0:min(25//((depth+1)**2), len(argList))]

        if action:
            return val, action

        if potential:
            return val, potential

        for index, value in argList:

            if timeStart:
                gap = time.time() - timeStart
                if gap > 13:
                    break

            board.addP1(index)
            board._updateFrontier(index)
            board.value = value

            moveVal, moveAction = self.minValue(board, alpha, beta, depth + 1)

            # if depth == 0:
            #     print(index, moveVal)


            board.value = valLocal
            board.frontier = set(frontierLocal)
            board.remove(index)

            if moveVal > val:
                val = moveVal
                action = index

            if val > beta:
                return val, action

            alpha = max(alpha, val)

        if not action:
            val = board.value[0]

        return val, action

    def minValue(self, board, alpha, beta, depth, ind=None):
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
            return board.value[1], None

        val = float('inf')

        action = None
        frontierLocal = tuple(board.frontier)
        valLocal = board.value

        argList = []
        # calculate the current value
        for index in frontierLocal:
            valP1, valP2 = valLocal
            dvalP1, dvalP2 = board.localUtility(index)
            valP1 -= dvalP1
            valP2 -= dvalP2

            board.addP2(index)
            dvalP1, dvalP2 = board.localUtility(index)
            if abs(valP1) > 500000 or abs(valP2) > 500000:
                dvalP1 *= 0
                dvalP2 *= 0

            valP1 += dvalP1
            valP2 += dvalP2
            board.remove(index)

            value = (valP1, valP2)

            argList.append((index, value))

        if depth + 1 > self.MAX_DEPTH:
            sum = 0
            for index, value in argList:
                sum += value[0]
            return sum/len(argList), action

        argList.sort(key=lambda x: x[1][0])
        if depth > 0:
            argList = argList[0:min(len(argList), 15//((depth+1)**2))]

        for index, value in argList:

            board.addP2(index)
            board._updateFrontier(index)
            board.value = value

            moveVal, moveAction = self.maxValue(board, alpha, beta, depth + 1)

            board.value = valLocal
            board.frontier = set(frontierLocal)
            board.remove(index)

            if moveVal < val:
                val = moveVal
                action = index

            if val < alpha:
                return val, action

            beta = min(beta, val)

        if not action:
            val = board.value[1]

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
        timeStart = time.time()
        val, action = self.maxValue(board, float('-inf'), float('inf'), depth, timeStart)
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
    filename = 'C:/Users/Daniel/Desktop/Daniel/projects/Gomoku/code/pydan2/board4.json'
    with open(filename, 'r') as f:
        b = json.load(f)
    board = Board(board=b)
    board._display()
    ai = AIPlayer()
    print(ai.getMove(board))

