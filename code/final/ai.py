import time
import re
import json
import pprint
import random


class Zobrist(object):

    def __init__(self):
        self.hashP1 = [[random.randint(0, 9223372036854775807) for _ in range(20)] for _ in range(20)]
        self.hashP2 = [[random.randint(0, 9223372036854775807) for _ in range(20)] for _ in range(20)]
        self.hashBoard = random.randint(0, 9223372036854775807)
        self.count = 0

    def move(self, index, player):
        x, y = index
        self.count += 1
        if player == P1:
            self.hashBoard ^= self.hashP1[x][y]
        elif player == P2:
            self.hashBoard ^= self.hashP2[x][y]
        else:
            assert 0, "empty do not take hashing move"

    def demove(self, index, player):
        x, y = index
        self.count -= 1
        if player == P1:
            self.hashBoard ^= self.hashP1[x][y]
        elif player == P2:
            self.hashBoard ^= self.hashP2[x][y]
        else:
            assert 0, "empty do not take hashing move"


def clearCache(level, cache: dict):
    cacheCopy = dict()
    for count, hashing in cache.keys():
        if count > level:
            cacheCopy[(count, hashing)] = cache[(count, hashing)]
    return cacheCopy


OUT = -1
BLANK = 0
P1 = 1
P2 = 2

CANDIDATES = [8, 8, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2]
CANDIDATES_THREATS_SEARCH = [10000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
MAX_SEARCH = len(CANDIDATES) - 1
MAX_DEPTH = 1
MAX_TIME = 3
K = 2
PENALTY = -1                # Conservative or aggressive?
REWARD = 1                  # ?
DISCOUNT1 = 1               # Difference between cases
DISCOUNT2 = 1               # Difference between cases
DISCOUNT3_P1 = 0.1          # On offensive or defensive?
DISCOUNT3_P2 = 0.1

SCORE_P1 = {
    # Winning Threats
    "FIVE": 1000000,                                     # ooooo
    "OPEN_FOUR": 100000,                                 # _oooo_
    "COMPENSATION_FOUR": 100000 - 2 * 1000 * REWARD,     # o_ooo_o
    # Forcing Threats
    "SIMPLE_FOUR": 1000 * REWARD,                        # xoooo_ / oo_oo / ooo_o
    "OPEN_THREE": 1000,                                  # __ooo__
    "SIMPLE_THREE": 1000 * DISCOUNT1,                    # _oo_o_ / x_ooo__
    "COMPENSATION": 1000 - 50,                           # o_o_o_o_o -> 3 * o_o_o + 630 = 1000
    # Potential Threats
    "OPEN_TWO": 40,                                      # __oo__ / __o_o__
    "BROKEN_THREE": 50,                                  # xooo__ / xoo_o_ / xo_oo_ / x_ooo_x / o_o_o
    "SIMPLE_TWO": 10 * DISCOUNT2,                        # x_oo__ / x_o_o__
}

THREATS_P1 = {
    # Winning Threats
    "FIVE": re.compile(r'11111'),
    "OPEN_FOUR": re.compile(r'011110'),
    "COMPENSATION_FOUR": re.compile(r'1011101'),
    # Forcing Threats
    "SIMPLE_FOUR": re.compile(r'[23]11110|01111[23]|11011|11101|10111'),
    "OPEN_THREE": re.compile(r'0011100'),
    "SIMPLE_THREE": re.compile(r'011010|010110|[23]011100|001110[23]'),
    "COMPENSATION": re.compile(r'101010101'),
    # Potential Threats
    "OPEN_TWO": re.compile(r'001100|0010100'),
    "BROKEN_THREE": re.compile(r'[23]11100|00111[23]|[23]11010|01011[23]|[23]10110|01101[23]|[23]01110[23]|10101'),
    "SIMPLE_TWO": re.compile(r'[23]01100|00110[23]|[23]010100|001010[23]'),
}

SCORE_P2 = {
    # Winning Threats
    "FIVE": 1000000 * -1,
    "OPEN_FOUR": 100000 * -1,
    "COMPENSATION_FOUR": (100000 - 2 * 1000 * REWARD * PENALTY) * -1,
    # Forcing Threats
    "SIMPLE_FOUR": 1000 * REWARD * PENALTY,
    "OPEN_THREE": 1000 * PENALTY,
    "SIMPLE_THREE": 1000 * DISCOUNT1 * PENALTY,
    "COMPENSATION": (1000 - 50) * PENALTY,
    # Potential Threats
    "OPEN_TWO": 40 * PENALTY,
    "BROKEN_THREE": 50 * PENALTY,
    "SIMPLE_TWO": 10 * DISCOUNT2 * PENALTY,
}

THREATS_P2 = {
    # Winning Threats
    "FIVE": re.compile(r'22222'),
    "OPEN_FOUR": re.compile(r'022220'),
    "COMPENSATION_FOUR": re.compile(r'2022202'),
    # Forcing Threats
    "SIMPLE_FOUR": re.compile(r'[13]22220|02222[13]|22022|22202|20222'),
    "OPEN_THREE": re.compile(r'0022200'),
    "SIMPLE_THREE": re.compile(r'022020|020220|[13]022200|002220[13]'),
    "COMPENSATION": re.compile(r'202020202'),
    # Potential Threats
    "OPEN_TWO": re.compile(r'002200|0020200'),
    "BROKEN_THREE": re.compile(r'[13]22200|00222[13]|[13]22020|02022[13]|[13]20220|02202[13]|[13]02220[13]|20202'),
    "SIMPLE_TWO": re.compile(r'[13]02200|00220[13]|[13]020200|002020[13]'),
}

zobrist = Zobrist()
stateCache = dict()
hashBoard = zobrist.hashBoard


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
    def __init__(self, width=0, height=0, board=None):  # k: kNN

        self.k = K
        self.frontier = set()
        if board:
            self.width = len(board[0])
            self.height = len(board)
            self.board = board

            # Make zobrist
            zobrist.hashBoard = hashBoard
            zobrist.count = 0
            for x in range(self.width):
                for y in range(self.height):
                    if self[x, y] == P1:
                        zobrist.move((x, y), P1)
                    if self[x, y] == P2:
                        zobrist.move((x, y), P2)

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

        # Initialize threat space
        self.threatsP1 = {key: 0 for key in SCORE_P1.keys()}
        self.threatsP2 = {key: 0 for key in SCORE_P2.keys()}
        self._threatSearch()

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

    def count(self):
        count = 0
        for x in range(self.width):
            for y in range(self.height):
                if self[x, y] > 0:
                    count += 1
        return count

    def remove(self, index):
        """
            Not a decent method!
        """
        x, y = index
        self.board[x][y] = 0

    def evaluate(self, player, method='linear'):
        """
        Evaluate the board for its utility.

        Args:
            player: P1 or P2
            method: 'linear' or 'q-learning'

        Returns:
            score: the final evaluation of the board
        """
        if method == 'linear':
            discountP1 = 1
            discountP2 = 1
            if player == P1:
                discountP1 = DISCOUNT3_P1
                if self.threatsP1["FIVE"]:
                    return SCORE_P1["FIVE"]
                if self.threatsP2["FIVE"]:
                    return SCORE_P2["FIVE"]
                if self.threatsP2["OPEN_FOUR"] or self.threatsP2["COMPENSATION_FOUR"]:
                    return SCORE_P2["OPEN_FOUR"]
                if self.threatsP1["OPEN_FOUR"] or self.threatsP1["COMPENSATION_FOUR"] or \
                    self.threatsP1["SIMPLE_FOUR"] + self.threatsP1["OPEN_THREE"] + self.threatsP1["SIMPLE_THREE"] + \
                        self.threatsP1["COMPENSATION"] > 1:
                    return SCORE_P1["OPEN_FOUR"]
                if self.threatsP2["SIMPLE_FOUR"] or self.threatsP2["OPEN_THREE"] or self.threatsP2["SIMPLE_THREE"] or \
                        self.threatsP2["COMPENSATION"]:
                    return SCORE_P2["OPEN_THREE"]

                score = 0
                for pattern in self.threatsP1.keys():
                    score += self.threatsP1[pattern] * SCORE_P1[pattern] * discountP1 \
                             + self.threatsP2[pattern] * SCORE_P2[pattern] * discountP2
                return score

            else:
                discountP2 = DISCOUNT3_P2
                if self.threatsP2["FIVE"]:
                    return SCORE_P2["FIVE"]
                if self.threatsP1["FIVE"]:
                    return SCORE_P1["FIVE"]
                if self.threatsP1["OPEN_FOUR"] or self.threatsP1["COMPENSATION_FOUR"]:
                    return SCORE_P1["OPEN_FOUR"]
                if self.threatsP2["OPEN_FOUR"] or self.threatsP2["COMPENSATION_FOUR"] or \
                    self.threatsP2["SIMPLE_FOUR"] + self.threatsP2["OPEN_THREE"] + self.threatsP2["SIMPLE_THREE"] + \
                        self.threatsP2["COMPENSATION"] > 1:
                    return SCORE_P2["OPEN_FOUR"]
                if self.threatsP1["SIMPLE_FOUR"] or self.threatsP1["OPEN_THREE"] or self.threatsP1["SIMPLE_THREE"] or \
                        self.threatsP1["COMPENSATION"]:
                    return SCORE_P1["OPEN_THREE"]

                score = 0
                for pattern in self.threatsP1.keys():
                    score += self.threatsP1[pattern] * SCORE_P1[pattern] * discountP1 \
                             + self.threatsP2[pattern] * SCORE_P2[pattern] * discountP2
                score = 0
                for pattern in self.threatsP1.keys():
                    score += self.threatsP1[pattern] * SCORE_P1[pattern] * discountP1 \
                             + self.threatsP2[pattern] * SCORE_P2[pattern] * discountP2
                return score

        elif method == 'q-learning':
            raise NotImplemented
        else:
            raise NotImplemented

    def localThreats(self, lastMove, display=False):
        """
        Get the local difference caused by last move.

        Args:
            lastMove: (x, y) index of last move.

        Returns:
            localThreatsP1: dict
            localThreatsP2: dict
        """
        localThreatsP1 = {key: 0 for key in SCORE_P1.keys()}
        localThreatsP2 = {key: 0 for key in SCORE_P2.keys()}
        x, y = lastMove

        # Four directions of (x, y) that may involve
        h = ''.join(map(str, [self.board[x][i] for i in range(0, self.height)]))
        v = ''.join(map(str, [self.board[i][y] for i in range(0, self.height)]))
        d1 = ''.join(
            map(str, [self.board[x + i][y + i] for i in range(-min([x, y]), self.width - max(x, y))]))
        d2 = ''.join(map(str, [self.board[x + i][y - i] for i in
                               range(-min([x, self.width - y - 1]), min([self.width - x, y + 1]))]))
        s = '3'+h+'3'+v+'3'+d1+'3'+d2+'3'

        if display:
            print()

        for key, pattern in THREATS_P1.items():
            localThreatsP1[key] += len(re.findall(pattern, s))
            if display:
                print(key, re.findall(pattern, s))
        for key, pattern in THREATS_P2.items():
            localThreatsP2[key] += len(re.findall(pattern, s))
            if display:
                print(key, re.findall(pattern, s))

        return localThreatsP1, localThreatsP2

    def toJson(self, filename):
        with open(filename, 'w') as f:
            f.write(pprint.pformat(self.board))

    def loadJson(self, filename):
        with open(filename, 'r') as f:
            board = json.load(f)
            self.width = len(board[0])
            self.height = len(board)
            self.board = board

            # Make zobrist
            zobrist.hashBoard = hashBoard
            zobrist.count = 0
            for x in range(self.width):
                for y in range(self.height):
                    if self[x, y] == P1:
                        zobrist.move((x, y), P1)
                    if self[x, y] == P2:
                        zobrist.move((x, y), P2)

            # Make frontier
            for x in range(self.height):
                for y in range(self.width):
                    if self[x, y] > 0:
                        for i in range(1, self.k + 1):
                            for neighbour in kNN(x, y, i):
                                if self[neighbour] == BLANK:
                                    self.frontier.add(neighbour)

            # Initialize threat space
            self.threatsP1 = {key: 0 for key in SCORE_P1.keys()}
            self.threatsP2 = {key: 0 for key in SCORE_P2.keys()}
            self._threatSearch()

    def _updateFrontier(self, index):
        """
            Local update for each piece of node.
        """
        x, y = index
        if (x, y) in self.frontier:
            self.frontier.remove((x, y))
        for i in range(1, self.k + 1):
            for neighbour in kNN(x, y, i):
                if self[neighbour] == BLANK:
                    self.frontier.add(neighbour)

    def _display(self, f=False):
        # TODO: Visualize it
        print(f'   {" ".join(("%2d" %i for i in range(self.width)))}')
        display = [['-' for __ in range(self.width)] for _ in range(self.height)]
        for x in range(self.height):
            for y in range(self.width):
                if self[x, y] == 1:
                    display[x][y] = str(P1)
                elif self[x, y] == 2:
                    display[x][y] = str(P2)
                elif (x, y) in self.frontier and f:
                    display[x][y] = 'f'
        for i, row in enumerate(display):
            print(f'{"%-2d"%i} {"  ".join(row)}')

    def _threatSearch(self, display=False):
        """
            A private method to initialize the threat space.
        """

        self.threatsP1 = {key: 0 for key in SCORE_P1.keys()}
        self.threatsP2 = {key: 0 for key in SCORE_P2.keys()}
        s = self._toString()

        if display:
            self._display()

        for key, pattern in THREATS_P1.items():
            if display:
                print(key, re.findall(pattern, s))
            self.threatsP1[key] += len(re.findall(pattern, s))
        for key, pattern in THREATS_P2.items():
            if display:
                print(key, re.findall(pattern, s))
            self.threatsP2[key] += len(re.findall(pattern, s))

    def _toString(self):
        """
            A private method to stringify the whole board.
        """

        h = '3'.join((''.join((str(_) for _ in line)) for line in self.board))
        v = '3'.join((''.join((str(self.board[i][j]) for i in range(self.width))) for j in range(self.height)))

        d1 = '3'.join((''.join((str(self.board[self.width - p + q - 1][q])
                                 for q in range(max(p - self.width + 1, 0), min(p + 1, self.height))))
                        for p in range(self.width + self.height - 1)))
        d2 = '3'.join((''.join((str(self.board[p - q][q])
                                 for q in range(max(p - self.width + 1, 0), min(p + 1, self.height))))
                        for p in range(self.width + self.height - 1)))
        return '3'+h+'3'+v+'3'+d1+'3'+d2+'3'


class AIPlayer(object):
    def __init__(self):
        self.p1 = P1
        self.p2 = P2

    def terminalTest(self, depth, candidate):
        if depth > MAX_DEPTH or candidate > MAX_SEARCH:
            return True

    def maxValue(self, board: Board, alpha, beta, depth, candidate, timeStart=None):
        """
        Alpha-beta search for max nodes

        Args:
            board : Board() object
            depth : current depth
            timeStart (Only for depth 0): timestamp to avoid overtime
        Returns:
            v: the utility of the index
            action: an index (x, y)
        """
        val = float('-inf')
        action = None

        # Exploration first
        frontierLocal = tuple(board.frontier)
        threatsP1Copy = tuple(board.threatsP1.items())
        threatsP2Copy = tuple(board.threatsP2.items())
        argList, killP1, potentP1, killP2, potentP2 = self.exploreP1(board, threatsP1Copy,
                                                                     threatsP2Copy, frontierLocal, depth)

        # Early decision
        if killP1:
            return SCORE_P1["FIVE"], killP1[0]
        elif killP2:
            argLocal = []
            for valP1, valP2, index, threatsP1, threatsP2 in argList:
                if index in killP2:
                    argLocal.append((valP1, valP2, index, threatsP1, threatsP2))
            argList = argLocal
            depth = max(depth - 2, 0)
        elif potentP1:
            argLocal = []
            for valP1, valP2, index, threatsP1, threatsP2 in argList:
                if index in potentP1:
                    argLocal.append((valP1, valP2, index, threatsP1, threatsP2))
            argList = argLocal
            depth = max(depth - 2, 0)
        elif potentP2:
            argLocal = []
            for valP1, valP2, index, threatsP1, threatsP2 in argList:
                if index in potentP2:
                    argLocal.append((valP1, valP2, index, threatsP1, threatsP2))
            argList = argLocal
            depth = max(depth - 2, 0)
        argList.sort(key=lambda x: x[0], reverse=True)
        argList = argList[0:min(CANDIDATES[candidate], len(argList))]  # Sliced?

        if self.terminalTest(depth + 1, candidate + 1):
            return argList[0][0], argList[0][2]

        for valP1, valP2, index, threatsP1, threatsP2 in argList:
            # Timer
            if timeStart:
                gap = time.time() - timeStart
                if gap > MAX_TIME:
                    break

            board.addP1(index)
            zobrist.move(index, P1)
            board._updateFrontier(index)
            board.threatsP1 = dict(threatsP1)
            board.threatsP2 = dict(threatsP2)

            moveVal, moveAction = self.minValue(board, alpha, beta, depth + 1, candidate + 1)

            board.threatsP1 = dict(threatsP1Copy)
            board.threatsP2 = dict(threatsP2Copy)
            board.frontier = set(frontierLocal)
            board.remove(index)
            zobrist.demove(index, P1)

            if moveVal > val:
                val = moveVal
                action = index

            if val > beta:
                return val, action

            alpha = max(alpha, val)

        if not action:
            val = board.evaluate(P1)

        return val, action

    def minValue(self, board, alpha, beta, depth, candidate):
        """
        Alpha-beta search for min nodes

        Args:
            board : Board() object
            depth : current depth
        Returns:
            v: the utility of the index
            action: an index (x, y)
        """
        val = float('-inf')
        action = None

        # Exploration first
        frontierLocal = tuple(board.frontier)
        threatsP1Copy = tuple(board.threatsP1.items())
        threatsP2Copy = tuple(board.threatsP2.items())
        argList, killP1, potentP1, killP2, potentP2 = self.exploreP2(board, threatsP1Copy,
                                                                     threatsP2Copy, frontierLocal, depth)

        # Early decision
        if killP2:
            return SCORE_P2["FIVE"], killP2[0]
        elif killP1:
            argLocal = []
            for valP1, valP2, index, threatsP1, threatsP2 in argList:
                if index in killP1:
                    argLocal.append((valP1, valP2, index, threatsP1, threatsP2))
            argList = argLocal
            depth = max(depth - 2, 1)
        elif potentP2:
            argLocal = []
            for valP1, valP2, index, threatsP1, threatsP2 in argList:
                if index in potentP2:
                    argLocal.append((valP1, valP2, index, threatsP1, threatsP2))
            argList = argLocal
            depth = max(depth - 2, 1)
        elif potentP1:
            argLocal = []
            for valP1, valP2, index, threatsP1, threatsP2 in argList:
                if index in potentP1:
                    argLocal.append((valP1, valP2, index, threatsP1, threatsP2))
            argList = argLocal
            depth = max(depth - 2, 1)

        argList.sort(key=lambda x: x[1], reverse=False)
        argList = argList[0:min(CANDIDATES[candidate], len(argList))]  # Sliced?
        if self.terminalTest(depth + 1, candidate + 1):
            return argList[0][1], argList[0][2]

        for valP1, valP2, index, threatsP1, threatsP2 in argList:

            board.addP2(index)
            zobrist.move(index, P2)
            board._updateFrontier(index)
            board.threatsP1 = dict(threatsP1)
            board.threatsP2 = dict(threatsP2)

            moveVal, moveAction = self.maxValue(board, alpha, beta, depth + 1, candidate + 1)

            board.threatsP1 = dict(threatsP1Copy)
            board.threatsP2 = dict(threatsP2Copy)
            board.frontier = set(frontierLocal)
            board.remove(index)
            zobrist.demove(index, P2)

            if moveVal < val:
                val = moveVal
                action = index

            if val < alpha:
                return val, action

            beta = min(beta, val)

        if not action:
            val = board.evaluate(P2)

        return val, action

    def exploreP1(self, board: Board, threatsP1Copy: tuple, threatsP2Copy: tuple, frontierLocal: tuple, depth):

        argList = []
        killP1 = []
        potentP1 = []
        killP2 = []
        potentP2 = []
        defenseP2 = []
        flag = False

        # calculate the current value
        for index in frontierLocal:
            threatsP1 = dict(threatsP1Copy)
            threatsP2 = dict(threatsP2Copy)
            dThreatsP1, dThreatsP2 = board.localThreats(index)
            for pattern in threatsP1.keys():
                threatsP1[pattern] -= dThreatsP1[pattern]
                threatsP2[pattern] -= dThreatsP2[pattern]
                assert threatsP1[pattern] >= 0 and threatsP2[pattern] >= 0, print(dThreatsP1, '\n', dThreatsP2, '\n')

            # Try hashing for potential P2 chances
            zobrist.move(index, P2)
            if (zobrist.count, zobrist.hashBoard) in stateCache:
                threatsP1, threatsP2 = stateCache[(zobrist.count, zobrist.hashBoard)]
                threatsP1, threatsP2 = dict(threatsP1), dict(threatsP2)
            else:
                board.addP2(index)
                dThreatsP1, dThreatsP2 = board.localThreats(index)
                for pattern in threatsP1.keys():
                    threatsP1[pattern] += dThreatsP1[pattern]
                    threatsP2[pattern] += dThreatsP2[pattern]
                board.remove(index)
                stateCache[(zobrist.count, zobrist.hashBoard)] = tuple(threatsP1.items()), tuple(threatsP2.items())

            if threatsP2["FIVE"]:
                killP2.append(index)
            if threatsP2["OPEN_FOUR"] or threatsP2["COMPENSATION_FOUR"] or dThreatsP2["SIMPLE_FOUR"] + dThreatsP2["OPEN_THREE"] \
                    + dThreatsP2["SIMPLE_THREE"] + dThreatsP2["COMPENSATION"] > 1:
                flag = True
                potentP2.append(index)
            if dThreatsP2["SIMPLE_FOUR"] + dThreatsP2["OPEN_THREE"] \
                    + dThreatsP2["SIMPLE_THREE"] + dThreatsP2["COMPENSATION"]:
                defenseP2.append(index)

            zobrist.demove(index, P2)

            threatsP1 = dict(threatsP1Copy)
            threatsP2 = dict(threatsP2Copy)
            dThreatsP1, dThreatsP2 = board.localThreats(index)
            for pattern in threatsP1.keys():
                threatsP1[pattern] -= dThreatsP1[pattern]
                threatsP2[pattern] -= dThreatsP2[pattern]
                assert threatsP1[pattern] >= 0 and threatsP2[pattern] >= 0, print(dThreatsP1, '\n', dThreatsP2, '\n')

            # Try hashing for potential P1 chances
            zobrist.move(index, P1)
            if (zobrist.count, zobrist.hashBoard) in stateCache:
                threatsP1, threatsP2 = stateCache[(zobrist.count, zobrist.hashBoard)]
                threatsP1, threatsP2 = dict(threatsP1), dict(threatsP2)

            else:
                board.addP1(index)
                dThreatsP1, dThreatsP2 = board.localThreats(index)
                for pattern in threatsP1.keys():
                    threatsP1[pattern] += dThreatsP1[pattern]
                    threatsP2[pattern] += dThreatsP2[pattern]

                board.remove(index)
                stateCache[(zobrist.count, zobrist.hashBoard)] = tuple(threatsP1.items()), tuple(threatsP2.items())

            if threatsP1["FIVE"]:
                killP1.append(index)
            if threatsP1["OPEN_FOUR"] or threatsP1["COMPENSATION_FOUR"] or dThreatsP1["SIMPLE_FOUR"] + dThreatsP1["OPEN_THREE"] \
                    + dThreatsP1["SIMPLE_THREE"] + dThreatsP1["COMPENSATION"] > 1:
                potentP1.append(index)

            zobrist.demove(index, P1)

            board.threatsP1 = threatsP1
            board.threatsP2 = threatsP2
            valP1 = board.evaluate(P1)
            valP2 = board.evaluate(P2)

            argList.append((valP1, valP2, index, tuple(threatsP1.items()), tuple(threatsP2.items())))

        # If a double three is potential for P2, then any defense of an open three is plausible (No)
        # if flag:
        #     potentP2 += defenseP2
        return argList, tuple(killP1), tuple(potentP1), tuple(killP2), tuple(potentP2)

    def exploreP2(self, board: Board, threatsP1Copy: tuple, threatsP2Copy: tuple, frontierLocal: tuple, depth):

        argList = []
        killP1 = []
        potentP1 = []
        killP2 = []
        potentP2 = []
        defenseP1 = []
        flag = False

        # calculate the current value
        for index in frontierLocal:
            threatsP1 = dict(threatsP1Copy)
            threatsP2 = dict(threatsP2Copy)
            dThreatsP1, dThreatsP2 = board.localThreats(index)
            for pattern in threatsP1.keys():
                threatsP1[pattern] -= dThreatsP1[pattern]
                threatsP2[pattern] -= dThreatsP2[pattern]
                assert threatsP1[pattern] >= 0 and threatsP2[pattern] >= 0, print(dThreatsP1, '\n', dThreatsP2, '\n')

            # Try hashing for potential P1 chances
            zobrist.move(index, P1)
            if (zobrist.count, zobrist.hashBoard) in stateCache:
                threatsP1, threatsP2 = stateCache[(zobrist.count, zobrist.hashBoard)]
                threatsP1, threatsP2 = dict(threatsP1), dict(threatsP2)
            else:
                board.addP1(index)
                dThreatsP1, dThreatsP2 = board.localThreats(index)
                for pattern in threatsP1.keys():
                    threatsP1[pattern] += dThreatsP1[pattern]
                    threatsP2[pattern] += dThreatsP2[pattern]
                board.remove(index)
                stateCache[(zobrist.count, zobrist.hashBoard)] = tuple(threatsP1.items()), tuple(threatsP2.items())

            if threatsP1["FIVE"]:
                killP1.append(index)
            if threatsP1["OPEN_FOUR"] or threatsP1["COMPENSATION_FOUR"] or dThreatsP1["SIMPLE_FOUR"] + dThreatsP1[
                "OPEN_THREE"] \
                    + dThreatsP1["SIMPLE_THREE"] + dThreatsP1["COMPENSATION"] > 1:
                flag = True
                potentP1.append(index)
            if dThreatsP1["SIMPLE_FOUR"] + dThreatsP1["OPEN_THREE"] \
                    + dThreatsP1["SIMPLE_THREE"] + dThreatsP1["COMPENSATION"]:
                defenseP1.append(index)

            zobrist.demove(index, P1)
            threatsP1 = dict(threatsP1Copy)
            threatsP2 = dict(threatsP2Copy)
            dThreatsP1, dThreatsP2 = board.localThreats(index)
            for pattern in threatsP1.keys():
                threatsP1[pattern] -= dThreatsP1[pattern]
                threatsP2[pattern] -= dThreatsP2[pattern]
                assert threatsP1[pattern] >= 0 and threatsP2[pattern] >= 0, print(dThreatsP1, '\n', dThreatsP2,
                                                                                  '\n')

            # Try hashing for potential P2 chances
            zobrist.move(index, P2)
            if (zobrist.count, zobrist.hashBoard) in stateCache:
                threatsP1, threatsP2 = stateCache[(zobrist.count, zobrist.hashBoard)]
                threatsP1, threatsP2 = dict(threatsP1), dict(threatsP2)
            else:
                board.addP2(index)
                dThreatsP1, dThreatsP2 = board.localThreats(index)
                for pattern in threatsP1.keys():
                    threatsP1[pattern] += dThreatsP1[pattern]
                    threatsP2[pattern] += dThreatsP2[pattern]
                board.remove(index)
                stateCache[(zobrist.count, zobrist.hashBoard)] = tuple(threatsP1.items()), tuple(threatsP2.items())

            if threatsP2["FIVE"]:
                killP2.append(index)
            if threatsP2["OPEN_FOUR"] or threatsP2["COMPENSATION_FOUR"] or dThreatsP2["SIMPLE_FOUR"] + dThreatsP2[
                "OPEN_THREE"] \
                    + dThreatsP2["SIMPLE_THREE"] + dThreatsP2["COMPENSATION"] > 1:
                potentP2.append(index)

            zobrist.demove(index, P2)

            board.threatsP1 = threatsP1
            board.threatsP2 = threatsP2
            valP1 = board.evaluate(P1)
            valP2 = board.evaluate(P2)

            argList.append((valP1, valP2, index, tuple(threatsP1.items()), tuple(threatsP2.items())))

        # If a double three is potential for P1, then any defense of an open three is plausible (No)
        # if flag:
        #     potentP1 += defenseP1
        return argList, tuple(killP1), tuple(potentP1), tuple(killP2), tuple(potentP2)

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
        candidate = 0
        timeStart = time.time()
        val, action = self.maxValue(board, float('-inf'), float('inf'), depth, candidate, timeStart)
        return val, action

    def getMove(self, board):
        global stateCache
        # stateCache = clearCache(zobrist.count, stateCache)
        _, action = self.alphaBetaSearch(board)
        return action


def trans90(board):
    board = board[::-1]
    rows, cols = len(board), len(board[0])
    for i in range(rows):
        for j in range(i, cols):
            if i == j:
                continue
            else:
                board[i][j], board[j][i] = board[j][i], board[i][j]
    return board


def fold(board):
    board = board[::-1]
    return board


if __name__ == "__main__":
    filename = 'C:/Users/Daniel/Desktop/Daniel/projects/Gomoku/code/final2/board1.json'
    with open(filename, 'r') as f:
        b = json.load(f)

        for i in range(2):
            for j in range(4):
                ai = AIPlayer()
                board = Board(board=b)
                board._display()
                print(ai.getMove(board))
                print(len(stateCache))
            b = fold(b)
