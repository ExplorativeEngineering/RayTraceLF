import time

# Timer =====================
timer = time.time()
what =""

def startTime():
    global timer
    timer = time.time()

# def startTime(what_):
#     global timer
#     timer = time.time()
#     print("   (((  ", what_)
#     global what
#     what = what_

def endTime():
    global what
    print(what, " = %s sec" % (time.time() - timer))
    what = "-"

def endTime(what_):
    global what
    what = what_
    print(what, " = %s sec" % (time.time() - timer))
    what = "-"

