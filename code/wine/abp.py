import random
import pisqpipe as pp
import win32api

WIN = 7
FLEX4 = 6
BLOCK4 = 5
FLEX3 = 4
BLOCK3 = 3
FLEX2 = 2
BLOCK2 = 1

ME = 1
OPP = 0

SIZE = 20
STOP = False
CacheSize = 1 << 22
pvsSize = 1 << 20
MAX_DEPTH = 6
MIN_DEPTH = 2

dx = [1, 0, 1, 1]
dy = [0, 1, 1, -1]

board_start = 0
board_end = 0
step = 0
zobristkey = 0
cnt = 1000
SearchDepth = 0


class Point(object):
    def __init__(self, val=0):
        self.pos = [0, 0]
        self.val = val


class Cell(object):
    def __init__(self):
        self.player = 0
        self.nei = 0
        self.pattern = [[0 for _ in range(4)] for _ in range(2)]


class Hash(object):
    def __init__(self, key=0, depth=0, state=0, val=0):
        self.key = key
        self.depth = depth
        self.state = state
        self.val = val


class PV(object):
    def __init__(self):
        self.key = 0
        self.best = [0, 0]


class MoveList(object):
    def __init__(self):
        self.phase = 0
        self.n = 0
        self.index = 0
        self.first = False
        self.hash_move = [0, 0]
        self.moves = [[0, 0] for _ in range(64)]


board = [[Cell() for _ in range(20 + 8)] for _ in range(20 + 8)]
moved = [[0, 0] for _ in range(20 * 20)]
cand = [Point() for _ in range(400)]
root_move = [Point() for _ in range(64)]
root_count = 0
hashtable = [Hash() for _ in range(CacheSize)]
pvstable = [PV() for _ in range(pvsSize)]
evaluate_table = [[[[0 for _ in range(3)] for _ in range(6)] for _ in range(6)] for _ in range(10)]
pattern_table = [[0 for _ in range(2)] for _ in range(65536)]
zobrist = [[[0 for _ in range(20 + 4)] for _ in range(20 + 4)] for _ in range(2)]
pval_table = [[[[0 for _ in range(8)] for _ in range(8)] for _ in range(8)] for _ in range(8)]
is_lost = [[False for _ in range(20 + 4)] for _ in range(20 + 4)]
BestMove = Point()


def initialize(_size):
    global SIZE, board_start, board_end, board
    SIZE = _size
    board_start, board_end = 4, SIZE + 4
    for i in range(20 + 8):
        for j in range(20 + 8):
            if i < 4 or i >= board_end or j < 4 or j >= board_end:
                board[i][j].player = 3
            else:
                board[i][j].player = 2
    InitEvaluateTable()
    InitPatternTable()
    InitZobrist()
    InitPval()


def InitEvaluateTable():
    global evaluate_table
    for i in range(10):
        for j in range(6):
            for k in range(6):
                for l in range(3):
                    evaluate_table[i][j][k][l] = EvaluateAssist(i, j, k, l)


def EvaluateAssist(length, dist, count, block):
    """
    Given the above data return the pattern
    length: Total length
    dist: distance inside
    count: length of chess in a row
    block: number of blocks
    """
    if length >= 5 and count > 1:
        if count == 5:
            return WIN
        if length > 5 and dist < 5 and block == 0:
            if count == 2:
                return FLEX2
            if count == 3:
                return FLEX3
            if count == 4:
                return FLEX4
        else:
            if count == 2:
                return BLOCK2
            if count == 3:
                return BLOCK3
            if count == 4:
                return BLOCK4
    return 0


def InitPatternTable():
    """
    Pattern table
    Storing the patterns
    """
    global pattern_table
    for i in range(65536):
        pattern_table[i][0] = LinePattern(0, i)
        pattern_table[i][1] = LinePattern(1, i)


def LinePattern(player, key):
    line_left = [0] * 9
    line_right = [0] * 9

    for i in range(9):
        if i == 4:
            line_left[i] = player
            line_right[i] = player
        else:
            line_left[i] = key & 3
            line_right[8 - i] = key & 3
            key >>= 2

    p1 = ScanLine(line_left)
    p2 = ScanLine(line_right)

    if p1 == BLOCK3 and p2 == BLOCK3:
        for i in range(9):
            if line_left[i] == 2:
                line_left[i] = player
                five = CountFive(line_left, player)
                line_left[i] = 2
                if five >= 2:
                    return FLEX3
        return BLOCK3


    elif p1 == BLOCK4 and p2 == BLOCK4:
        five = CountFive(line_left, player)
        return FLEX4 if five >= 2 else BLOCK4


    else:
        return p1 if p1 > p2 else p2


def ScanLine(line):
    empty = 0
    block = 0
    length = 1
    dist = 1
    count = 1
    myPiece = line[4]
    for k in range(5, 9):
        if line[k] == myPiece:
            if empty + count > 4:
                break
            count += 1
            length += 1
            dist = empty + count
        elif line[k] == 2:
            length += 1
            empty += 1
        else:
            if line[k - 1] == myPiece:
                block += 1
            break
    empty = dist - count
    for k in range(3, -1, -1):
        if line[k] == myPiece:
            if empty + count > 4:
                break
            count += 1
            length += 1
            dist = empty + count
        elif line[k] == 2:
            length += 1
            empty += 1
        else:
            if line[k + 1] == myPiece:
                block += 1
            break
    return evaluate_table[length][dist][count][block]


def CountFive(line, player):
    five = 0
    for i in range(9):
        if line[i] == 2:
            count = 0
            j = i - 1
            while j >= 0 and line[j] == player:
                count += 1
                j -= 1
            j = i + 1
            while j <= 8 and line[j] == player:
                count += 1
                j += 1
            if count >= 4:
                five += 1
    return five


def InitZobrist():
    global zobrist
    for i in range(20 + 4):
        for j in range(20 + 4):
            zobrist[0][i][j] = random.getrandbits(64)
            zobrist[1][i][j] = random.getrandbits(64)


def InitPval():
    global pval_table
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    pval_table[i][j][k][l] = GetPval(i, j, k, l)


def GetPval(a, b, c, d):
    type = [0] * 8
    type[a] += 1
    type[b] += 1
    type[c] += 1
    type[d] += 1

    if type[WIN] > 0:
        return 5000

    if type[FLEX4] > 0 or type[BLOCK4] > 1:
        return 1200

    if type[BLOCK4] > 0 and type[FLEX4] > 0:
        return 1000

    if type[FLEX3] > 1:
        return 200

    val = [0, 2, 5, 5, 12, 12]
    score = 0
    for i in range(1, BLOCK4 + 1):
        score += val[i] * type[i]

    return score


def GetTime():
    return win32api.GetTickCount() - pp.start_time


def StopTime():
    return min(pp.info_timeout_turn, pp.info_time_left / 7)


def SearchHash(depth, alpha, beta):
    hash_cur = hashtable[zobristkey & (CacheSize - 1)]
    if hash_cur.key == zobristkey:
        if hash_cur.depth >= depth:
            if hash_cur.state == 0:
                return hash_cur.val
            elif hash_cur.state == 1 and hash_cur.val <= alpha:
                return hash_cur.val
            elif hash_cur.state == 2 and hash_cur.val >= beta:
                return hash_cur.val
    return -20000


def RecordHash(depth, val, state):
    global hashtable, zobristkey
    hashtable[zobristkey & (CacheSize - 1)] = Hash(zobristkey, depth, state, val)


def UpdatePattern(x, y):
    for i in range(4):
        for j in range(-4, 5):
            if j == 0:
                continue
            a, b = x + j * dx[i], y + j * dy[i]
            if board[a][b].player != 3:
                key = GetKey(a, b, i)
                board[a][b].pattern[0][i] = pattern_table[key][0]
                board[a][b].pattern[1][i] = pattern_table[key][1]


def GetKey(x, y, i):
    step_x = dx[i]
    step_y = dy[i]
    key = (board[x - step_x * 4][y - step_y * 4].player) ^ \
          (board[x - step_x * 3][y - step_y * 3].player << 2) ^ \
          (board[x - step_x * 2][y - step_y * 2].player << 4) ^ \
          (board[x - step_x * 1][y - step_y * 1].player << 6) ^ \
          (board[x + step_x * 1][y + step_y * 1].player << 8) ^ \
          (board[x + step_x * 2][y + step_y * 2].player << 10) ^ \
          (board[x + step_x * 3][y + step_y * 3].player << 12) ^ \
          (board[x + step_x * 4][y + step_y * 4].player << 14)
    return key


def move(next):
    global ME, OPP, board, zobristkey, zobrist, step
    x, y = next
    board[x][y].player = ME
    zobristkey ^= zobrist[ME][x][y]
    ME ^= 1
    OPP ^= 1
    moved[step] = next
    step += 1

    UpdatePattern(x, y)
    for i in range(x - 2, x + 3):
        for j in range(y - 2, y + 3):
            board[i][j].nei += 1


def remove():
    global ME, OPP, board, zobristkey, zobrist, step
    step -= 1
    x, y = moved[step]
    ME ^= 1
    OPP ^= 1
    board[x][y].player = 2
    zobristkey ^= zobrist[ME][x][y]

    UpdatePattern(x, y)
    for i in range(x - 2, x + 3):
        for j in range(y - 2, y + 3):
            board[i][j].nei -= 1


def restart():
    global hashtable, pvstable, step
    hashtable = [Hash() for _ in range(CacheSize)]
    pvstable = [PV() for _ in range(pvsSize)]
    while step:
        remove()


def PutChess(next):
    """
    As we should put chess in brain_my and brain_opp on real board,
    this is a tiny transform for this process
    """
    x, y = next
    move([x + 4, y + 4])


def GetBestMove():
    best = MainSearch()[:]
    return [best[0] - 4, best[1] - 4]


def MainSearch():
    global SearchDepth, STOP, is_lost, BestMove, SIZE
    best_move = [0, 0]

    if step == 0:
        best_move[0] = int(SIZE / 2) + 4
        best_move[1] = int(SIZE / 2) + 4
        return best_move
    if step <= 2:
        x = moved[0][0] + random.randint(0, step * 2) - step
        y = moved[0][1] + random.randint(0, step * 2) - step
        while board[x][y].player == 3 or board[x][y].player != 2:
            x = moved[0][0] + random.randint(0, step * 2) - step
            y = moved[0][1] + random.randint(0, step * 2) - step
        return [x, y]

    STOP = False
    BestMove.val = 0
    is_lost = [[False for _ in range(20 + 4)] for _ in range(20 + 4)]

    for i in range(MIN_DEPTH, MAX_DEPTH + 1, 2):
        if STOP:
            break
        SearchDepth = i
        BestMove = RootSearch(SearchDepth, -10001, 10000)
        if STOP or (SearchDepth >= 10 and GetTime() >= 1000 and GetTime() * 12 > StopTime()):
            break
    best_move = BestMove.pos[:]
    return best_move


def RootSearch(depth, alpha, beta):
    global root_count, is_lost, root_move, STOP
    best = Point(val=root_move[0].val)
    best.pos = root_move[0].pos[:]

    if depth == MIN_DEPTH:
        moves = [[0, 0] for _ in range(64)]
        root_count = GetMoves(moves)
        if root_count == 1:
            STOP = True
            best.pos = moves[0]
            best.val = 0
            return best

        for i in range(root_count):
            root_move[i].pos = moves[i][:]
    else:
        for i in range(root_count):
            if root_move[i].val > root_move[0].val:
                root_move[0], root_move[i] = root_move[i], root_move[0]

    val = 0
    update_best = False
    for i in range(root_count):
        p = root_move[i].pos[:]
        if not is_lost[p[0]][p[1]]:
            move(p)
            for _ in range(1):
                if i > 0 and alpha + 1 < beta:
                    val = -AlphaBeta(depth - 1, -alpha - 1, -alpha)
                    if val <= alpha or val >= beta:
                        break
                val = -AlphaBeta(depth - 1, -beta, -alpha)
            remove()

            root_move[i].val = val

            if STOP:
                break

            if val == -10000:
                is_lost[p[0]][p[1]] = True

            if val > alpha:
                alpha = val
                best.pos = p[:]
                best.val = val
                update_best = True

                if val == 10000:
                    STOP = True
                    return best

    return best if update_best else root_move[0]


def AlphaBeta(depth, alpha, beta):
    global STOP
    global cnt
    cnt -= 1
    if cnt <= 0:
        cnt = 1000
        if GetTime() + 50 >= StopTime():
            STOP = True
            return alpha

    c = board[moved[step - 1][0]][moved[step - 1][1]]
    if c.pattern[OPP][0] == WIN or c.pattern[OPP][1] == WIN or c.pattern[OPP][2] == WIN or c.pattern[OPP][3] == WIN:
        return -10000

    if depth <= 0:
        return evaluateBoard()

    val = SearchHash(depth, alpha, beta)
    if val != -20000:
        return val

    my_moves = MoveList()
    my_moves.phase = 0
    my_moves.first = True
    p = NextMove(my_moves)
    best = Point()
    best.pos = p[:]
    best.val = -10000
    state = 1
    while p[0] != -1:
        move(p)
        for _ in range(1):
            if (not my_moves.first) and (alpha + 1 < beta):
                val = -AlphaBeta(depth - 1, -alpha - 1, -alpha)
                if val <= alpha or val >= beta:
                    break
            val = -AlphaBeta(depth - 1, -beta, -alpha)
        remove()

        if STOP:
            return best.val

        if val >= beta:
            RecordHash(depth, val, 2)
            RecordPVS(p)
            return val
        if val > best.val:
            best.val = val
            best.pos = p[:]
            if val > alpha:
                state = 0
                alpha = val
        p = NextMove(my_moves)
        my_moves.first = False

    RecordHash(depth, best.val, state)
    RecordPVS(best.pos)

    return best.val


def RecordPVS(best):
    global pvstable
    pv = PV()
    pv.key = zobristkey
    pv.best = best[:]
    pvstable[zobristkey % pvsSize] = pv


def NextMove(my_moves):
    """
    Find next move
    different phase means different generating method
    phase 0: Stored move
    phase 1: Generate all moves
    phase 2: Return moves one by one
    """

    if my_moves.phase == 0:
        my_moves.phase = 1
        pv = pvstable[zobristkey % pvsSize]

        if pv.key == zobristkey:
            my_moves.hashMove = pv.best
            return pv.best

    if my_moves.phase == 1:
        my_moves.phase = 2
        my_moves.n = GetMoves(my_moves.moves)
        my_moves.index = 0
        if not my_moves.first:
            for i in range(my_moves.n):
                if my_moves.moves[i][0] == my_moves.hashMove[0] and my_moves.moves[i][1] == my_moves.hashMove[1]:
                    for j in range(i + 1, my_moves.n):
                        my_moves.moves[j - 1] = my_moves.moves[j]
                    my_moves.n -= 1
                    break

    if my_moves.phase == 2:
        if my_moves.index < my_moves.n:
            my_moves.index += 1
            return my_moves.moves[my_moves.index - 1]
    return [-1, -1]


def GetMoves(move):
    global cand
    cand_count = 0

    for i in range(board_start, board_end):
        for j in range(board_start, board_end):
            if board[i][j].nei and board[i][j].player == 2:
                val = pointScore(board[i][j])
                if val > 0:
                    cand[cand_count].pos[0] = i
                    cand[cand_count].pos[1] = j
                    cand[cand_count].val = val
                    cand_count += 1

    cand[0:cand_count] = sorted(cand[0:cand_count], key=lambda x: x.val, reverse=True)

    move_count = MovePruning(move, cand, cand_count)

    if move_count == 0:
        for i in range(cand_count):
            move[i] = cand[i].pos[:]
            move_count += 1
            if move_count >= 10:
                break
    return move_count


def MovePruning(move, cand, cand_count):
    if cand[0].val >= 2400:
        move[0] = cand[0].pos[:]
        return 1
    move_count = 0

    if cand[0].val == 1200:
        for i in range(cand_count):
            if cand[i].val == 1200:
                move[move_count] = cand[i].pos[:]
                move_count += 1
            else:
                break

        flag = False
        for i in range(move_count, cand_count):
            p = board[cand[i].pos[0]][cand[i].pos[1]]
            for k in range(4):
                if p.pattern[ME][k] == BLOCK4 or p.pattern[OPP][k] == BLOCK4:
                    flag = True
                    break
            if flag:
                move[move_count] = cand[i].pos[:]
                move_count += 1
                if move_count >= 10:
                    break

    return move_count


def pointScore(c):
    score = [0] * 2
    score[ME] = pval_table[c.pattern[ME][0]][c.pattern[ME][1]][c.pattern[ME][2]][c.pattern[ME][3]]
    score[OPP] = pval_table[c.pattern[OPP][0]][c.pattern[OPP][1]][c.pattern[OPP][2]][c.pattern[OPP][3]]

    if score[ME] >= 200 or score[OPP] >= 200:
        return score[ME] * 2 if score[ME] >= score[OPP] else score[OPP]

    else:
        return score[ME] * 2 + score[OPP]


def evaluateBoard():
    eval = [0, 2, 12, 18, 96, 144, 800, 1200]
    myType = [0] * 8
    oppType = [0] * 8

    for i in range(board_start, board_end):
        for j in range(board_start, board_end):
            if board[i][j].nei and board[i][j].player == 2:
                tmp_block4 = myType[BLOCK4]
                for k in range(4):
                    myType[board[i][j].pattern[ME][k]] += 1
                    oppType[board[i][j].pattern[OPP][k]] += 1

                if myType[BLOCK4] - tmp_block4 >= 2:
                    myType[BLOCK4] -= 2
                    myType[BLOCK4] += 1

    if myType[WIN] >= 1:
        return 10000

    if oppType[WIN] == 0 and myType[FLEX4] >= 1:
        return 10000

    if oppType[WIN] >= 2:
        return -10000

    my_score = 0
    opp_score = 0
    for i in range(8):
        my_score += myType[i] * eval[i]
        opp_score += oppType[i] * eval[i]

    return my_score * 2 - opp_score
