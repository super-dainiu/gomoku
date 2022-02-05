import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
from ai import Board, AIPlayer, MAX_DEPTH
import time

pp.infotext = 'name="pbrain-pydan1", author="Daniel", version="1.0", ' \
              'country="China", www="https://github.com/Tequila-Sunrise/FDU-Gomoku-Bot"'

MAX_BOARD = 20
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]


# TODO: AI

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

# define a file for logging ...
DEBUG_LOGFILE = "C:/Users/Daniel/Desktop/Daniel/projects/Gomoku/code/final2/tmp/zobrist.log"
# ...and clear it initially
with open(DEBUG_LOGFILE, "w+") as f:
    pass


# define a function for writing messages to the file
def logDebug(msg):
    with open(DEBUG_LOGFILE, "a") as f:
        f.write(msg + "\n")
        f.flush()


# define a function to get exception traceback
def logTraceBack():
    import traceback
    with open(DEBUG_LOGFILE, "a") as f:
        traceback.print_exc(file=f)
        f.flush()
    raise


# use logDebug wherever
# use try-except (with logTraceBack in except branch) to get exception info
# an example of problematic function
def brain_turn():
    try:
        myBoard = Board(board=board)
        myBoard.toJson('C:/Users/Daniel/Desktop/Daniel/projects/Gomoku/code/final2/board1.json')
        ai = AIPlayer()
        argList = {}
        t1 = time.time()
        MAX_DEPTH = 1
        while time.time() - t1 < 10:
            move = ai.getMove(myBoard)
            logDebug(f'{move}')
            MAX_DEPTH += 1
        x, y = move
        pp.do_mymove(x, y)
    except:
        logTraceBack()
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



######################################################################

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about


def main():
    pp.main()


if __name__ == "__main__":
    main()