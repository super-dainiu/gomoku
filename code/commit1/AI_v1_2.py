# __author:yaozhong  自动获取用户名
# date: 2021/11/11 自动获取时间
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import copy
import time
import math
import random

pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", www="https://github.com/stranskyjan/pbrain-pyrandom"'

MAX_BOARD = 20
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]
MaxDepth = 0 


def brain_init():
    if pp.width < 5 or pp.height < 5:
        pp.pipeOut("ERROR size of the board")
        return
    if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
        pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
        return
    pp.pipeOut("OK")


def brain_restart():
    for x in range(pp.width):
        for y in range(pp.height):
            board[x][y] = 0
    pp.pipeOut("OK")


def isFree(x, y):
    return x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] == 0


def brain_my(x, y):
    if isFree(x, y):
        board[x][y] = 1
    else:
        pp.pipeOut("ERROR my move [{},{}]".format(x, y))


def brain_opponents(x, y):
    if isFree(x, y):
        board[x][y] = 2
    else:
        pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))


def brain_block(x, y):
    if isFree(x, y):
        board[x][y] = 3
    else:
        pp.pipeOut("ERROR winning move [{},{}]".format(x, y))


def brain_takeback(x, y):
    if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
        board[x][y] = 0
        return 0
    return 2


def brain_turn():
    if pp.terminateAI:
        return
    i = 0
    while True:
        x = random.randint(0, pp.width)
        y = random.randint(0, pp.height)
        i += 1
        if pp.terminateAI:
            return
        if isFree(x, y):
            break
    if i > 1:
        pp.pipeOut("DEBUG {} coordinates didn't hit an empty field".format(i))
    pp.do_mymove(x, y)


case_dict = {'WIN': '11111', 'L4': '011110', 'S41': '011112', 'S42': '10111', 'S43': '0110110',
             'S45': '211011',
             'L31': '01110', 'L32': '010110', 'S31': '01112', 'S32': '010112', 'S33': '011012,', 'S34': '10011',
             'S35': '2011102',
             'L21': '00110', 'L22': '01010', 'L23': '010010', 'S21': '000112', 'S22': '001012', 'S23': '010012',
             "S24": '10001', 'S25': '2010102', 'S26': '2011002', 'L11': '00100'}


def num2str(list1):
    str1 = ''
    for i in range(len(list1)):
        str1 += str(list1[i])
    return str1


def str2num(str1):
    list1 = []
    for i in range(len(str1)):
        list1.append(str1[i])
    return list1


def patten_change(list1, player):
    list2 = []
    if player == 2:
        for i in range(len(list1)):
            if list1[i] != 0:
                list2.append(3 - int(list1[i]))
            else:
                list2.append(0)
    return num2str(list2)


def patten(list1, player):
    case_num_dict = {'WIN': 0, 'L4': 0, 'S41': 0, 'S42': 0, 'S43': 0, 
                     'S45': 0,
                     'L31': 0, 'L32': 0, 'S31': 0, 'S32': 0, 'S33': 0, 'S34': 0,
                     'S35': 0,
                     'L21': 0, 'L22': 0, 'L23': 0, 'S21': 0, 'S22': 0, 'S23': 0,
                     "S24": 0, 'S25': 0, 'S26': 0, 'L11': 0}
    if player == 1:
        for listi in list1:
            for casekey, casei in case_dict.items():
                list_find = num2str(listi)
                list_find1 = list_find[::-1]
                if list_find.find(casei) >= 0 or list_find1.find(casei) >= 0:
                    case_num_dict[casekey] += 1
                    break
    else:
        for listi in list1:
            for casekey, casei in case_dict.items():
                list_find = num2str(patten_change(listi, player))
                list_find1 = list_find[::-1]
                if list_find.find(casei) >= 0 or list_find1.find(casei) >= 0:
                    case_num_dict[casekey] += 1
                    break
    return case_num_dict


def find_list(board, x, y, player):
    # if board[x][y] != 0:
    #     print('error')
    #     return
    list1 = []
    Directionset = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for direction in Directionset:
        listi = []
        k = 0
        x_now1 = x_now2 = x
        y_now1 = y_now2 = y
        while (x_now1 >= 0 and y_now1 >= 0 and k <= 4 and x_now1 <= MAX_BOARD - 1 and y_now1 <= MAX_BOARD - 1):
            if x_now1 - direction[0] >= 0 and y_now1 - direction[1] >= 0 and y_now1 - direction[1] <= MAX_BOARD - 1:
                x_now1 = x_now1 - direction[0]
                y_now1 = y_now1 - direction[1]
                k += 1
            else:
                break
        k = 0
        while (x_now2 <= MAX_BOARD - 1 and y_now2 <= MAX_BOARD - 1 and k <= 4 and x_now2 >= 0 and y_now2 >= 0):
            if x_now2 + direction[0] <= MAX_BOARD - 1 and y_now2 + direction[1] >= 0 and y_now2 + direction[1] <= MAX_BOARD - 1:
                x_now2 = x_now2 + direction[0]
                y_now2 = y_now2 + direction[1]
                k += 1
            else:
                break
        for k in range(max(x_now2 - x_now1 + 1,abs(y_now2 - y_now1) + 1)):
            if x_now1 + k * direction[0] == x and y_now1 + k * direction[1] == y:
                listi.append(player)
            else:
                listi.append(board[x_now1 + k * direction[0]][y_now1 + k * direction[1]])
        list1.append(listi)
    return list1


def nearby(x, y, board, k1):
    Directionset = [(0, 1), (1, 0), (1, 1), (1, -1)]
    sum1 = 0

    for direction in Directionset:
        k = 0
        x_now1 = x_now2 = x
        y_now1 = y_now2 = y
        while (x_now1 >= 0 and y_now1 >= 0 and k <= 1 and x_now1 <= MAX_BOARD - 1 and y_now1 <= MAX_BOARD - 1):
            if x_now1 - direction[0] >= 0 and y_now1 - direction[1] >= 0 and y_now1 - direction[1] <= MAX_BOARD - 1:
                x_now1 = x_now1 - direction[0]
                y_now1 = y_now1 - direction[1]
                k += 1
            else:
                break
        k = 0
        while (x_now2 <= MAX_BOARD - 1 and y_now2 <= MAX_BOARD - 1 and k <= 1 and x_now2 >= 0 and y_now2 >= 0):
            if x_now2 + direction[0] <= MAX_BOARD - 1 and y_now2 + direction[1] >= 0 and y_now2 + direction[1] <= MAX_BOARD - 1:
                x_now2 = x_now2 + direction[0]
                y_now2 = y_now2 + direction[1]
                k += 1
            else:
                break
        for k in range(max(x_now2 - x_now1 + 1,abs(y_now2 - y_now1) + 1)):
            if x_now1 + k * direction[0] == x and y_now1 + k * direction[1] == y:
                continue
            else:
                if board[x_now1 + k * direction[0]][y_now1 + k * direction[1]] != 0:
                    sum1 += 1
    return sum1 >= k1


def find_successors(board, k=2):
    list1 = []
    for i in range(MAX_BOARD):
        for j in range(MAX_BOARD):
            if board[i][j] == 0 and nearby(i, j, board, k) is True:
                list1.append((i, j))
    return list1

def if_win(player,board1):
    # 1 means my AI wins, while 2 means the oppo-AI wins
    dict_kill={'myWin':None,'opWin':None,'myL4':None,'my43':None,'opL4':None,'op43':None,'my33':None,'op33':None}
    successors = find_successors(board1)
    for x,y in successors:
        dict1 = patten(find_list(board1, x, y, player), player)
        dict2 = patten(find_list(board1, x, y, 3 - player), 3 - player)
        if dict1['WIN'] :
            dict_kill['myWin'] =(x,y)   # my AI wins
        if dict2['WIN'] >= 1:
            dict_kill['opWin'] =(x,y)   # oppo-AI wins
        if dict1["L4"]:
            dict_kill['myL4'] = (x,y)
        if dict1["S41"] + dict1["S42"] + dict1["S43"] + dict1["S45"] >=1 and dict1["L31"] + dict1[
            "L32"] >= 1:
            dict_kill['my43']=(x,y)
        if dict2["L4"]:
            dict_kill['opL4'] = (x,y)
        if dict2["S41"] + dict2["S42"] + dict2["S43"] + dict2["S45"] >= 1 and dict2["L31"] + dict2[
            "L32"] >= 1:
            dict_kill['op43'] = (x,y)
        if dict1['L31'] + dict1['L32'] >=2:
            dict_kill['my33'] =(x,y)
        if dict2['L31'] + dict2['L32'] >= 2:
            dict_kill['op33'] = (x,y)
    for key,value in dict_kill.items():
        if value:
            return value
    return 0



def score(x, y, player, board1):
    dict1 = patten(find_list(board1, x, y, player), player)
    dict2 = patten(find_list(board1, x, y, 3-player), 3-player)
    dict_v1={'WIN': 0, 'L4': 0, 'S41': 50, 'S42': 50, 'S43': 50,
                     'S45': 50,
                     'L31': 300, 'L32': 300, 'S31': 10, 'S32': 10, 'S33': 10, 'S34': 10,
                     'S35': 10,
                     'L21': 150, 'L22': 150, 'L23': 50, 'S21': 2, 'S22': 2, 'S23': 2,
                     "S24": 2, 'S25': 2, 'S26': 2, 'L11': 5}
    dict_v2 = {'WIN': 0, 'L4': 0, 'S41': 5, 'S42': 5, 'S43': 5,
                      'S45': 5,
                      'L31': 50, 'L32': 50, 'S31': 5, 'S32': 5, 'S33': 5, 'S34': 5,
                      'S35': 5,
                      'L21': 5, 'L22': 5, 'L23': 5, 'S21': 1, 'S22': 1, 'S23': 1,
                      "S24": 1, 'S25': 1, 'S26': 1, 'L11': 0}
    value1 = value2 = 0
    for key1,num1 in dict1.items():
        value1 += num1* dict_v1[key1]
    for key2,num2 in dict2.items():
        value2 += num2* dict_v2[key2]

    if player == 1:
        return value1 + value2
    else:
        return -(value1 + value2)


def value(depth, player, board, position, alpha, beta):
    x, y = position
    if depth < MaxDepth:
        depth += 1
    else:
        return score(x, y, player, board)

    if player == 1:
        # player 2's turn
        return min_value(depth=depth, player=2, board=board, position=position, alpha=alpha, beta=beta)
    else:
        # player 1's turn
        return max_value(depth=depth, player=1, board=board, position=position, alpha=alpha, beta=beta)


def max_value(depth, player, board, position, alpha, beta):
    # 该步为player1 ，我方落子
    v = -math.inf
    successors = find_successors(board, k=2)
    for new_posi in successors:  # player 1 的可能走法
        x, y = new_posi
        board[x][y] = player
        v = max(v, value(depth=depth, player=1, board=board, position=new_posi, alpha=alpha, beta=beta))
        board[x][y] = 0
        alpha = max(alpha, v)
        if alpha >= beta:
            return v
    return v


def min_value(depth, player, board, position, alpha, beta):
    # 该步为player2 ，对方轮次
    v = math.inf
    successors = find_successors(board, k=2)
    for new_posi in successors:
        x, y = new_posi
        board[x][y] = player
        v = min(v, value(depth, 2, board, new_posi, alpha, beta))
        # traceback
        board[x][y] = 0
        beta = min(beta, v)
        if alpha >= beta:
            return v
    return v


def Mybrain_turn():
    sum1 = 0
    for i in range(MAX_BOARD):
        sum1 += sum(board[i])
    if 0 == sum1:
        pp.do_mymove(10, 10) # start game
        return
    if 2 == sum1:
        for i in range(MAX_BOARD):
            for j in range(MAX_BOARD):
                if 2 == board[i][j]:
                    pp.do_mymove(i+1,j)
                    return
    begin = time.time()
    newboard = []  # shrinkage
    for rowindex in range(pp.width):
        tmp = board[rowindex]
        newboard.append(copy.deepcopy(tmp[0:pp.height]))

    successor = find_successors(newboard, k=2)
    player = 1  # 假设是我方轮次
    best_position = None
    alpha0 = -200000  # 必输分数
    beta0 = 100000  # 必赢分数
    best_position = if_win(player,newboard)
    if best_position:
        x,y = best_position
        pp.do_mymove(x, y)
        return
    for position in successor:
        x1, y1 = position
        newboard[x1][y1] = player
        if len(successor) > 1:
            tmp_value = value(depth=0, player=1, alpha=alpha0, beta=beta0, board=newboard, position=position)
        else:
            best_position = position
            break
        newboard[x1][y1] = 0  # 还原
        if tmp_value >= 600:  # Win
            best_position = position
            break
        elif tmp_value > alpha0:
            best_position = position
            alpha0 = tmp_value

    x, y = best_position
    pp.do_mymove(x, y)


def brain_end():
    pass


def brain_about():
    pp.pipeOut(pp.infotext)


if DEBUG_EVAL:
    import win32gui


    def brain_eval(x, y):
        # TODO check if it works as expected
        wnd = win32gui.GetForegroundWindow()
        dc = win32gui.GetDC(wnd)
        rc = win32gui.GetClientRect(wnd)
        c = str(board[x][y])
        win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
        win32gui.ReleaseDC(wnd, dc)

######################################################################
# A possible way how to debug brains.
# To test it, just "uncomment" it (delete enclosing """)
######################################################################
"""
# define a file for logging ...
DEBUG_LOGFILE = "D:/大三上/人工智能/Final-PJ/pbrain-pyrandom-master/pbrain-pyrandom-master/pbrain-pyrandom.log"
# ...and clear it initially
with open(DEBUG_LOGFILE,"w") as f:
    pass

# define a function for writing messages to the file
def logDebug(msg):
    with open(DEBUG_LOGFILE,"a") as f:
        f.write(msg+"\n")
        f.flush()

# define a function to get exception traceback
def logTraceBack():
    import traceback
    with open(DEBUG_LOGFILE,"a") as f:
        traceback.print_exc(file=f)
        f.flush()
    raise
"""
# use logDebug wherever
# use try-except (with logTraceBack in except branch) to get exception info
# an example of problematic function
"""
def brain_turn():
    logDebug("some message 1")
    try:
        logDebug("some message 2")
        1. / 0. # some code raising an exception
        logDebug("some message 3") # not logged, as it is after error
    except:
        logTraceBack()
"""
######################################################################

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = Mybrain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
    pp.brain_eval = brain_eval


def main():
    pp.main()


if __name__ == "__main__":
    main()