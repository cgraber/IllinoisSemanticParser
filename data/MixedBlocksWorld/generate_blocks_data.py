IS_TRAIN_MODE = True

TRAIN_SIZE_SHAPES = 1400
TRAIN_SIZE_RANDOBJ = 0
TEST_SIZE_SHAPES = 600
TEST_SIZE_RANDOBJ = 0
maxNumShapes = 3

import random, sys, os, string, itertools
from random import randint, shuffle
import numpy as np

COMMANDS = ["Create", "Construct", "Build", "Form"]
CONNECTORS = ["and", "with"]
NEXT = ["Next", "Then", "After that", "Now"]

ROW = 0
COL = 1
SQUARE = 2
RECT = 3

if IS_TRAIN_MODE:
    GRID_WIDTH = 100
    GRID_HEIGHT = 100
else:
    GRID_WIDTH = 10 
    GRID_HEIGHT = 10

TOP = 0
LEFT = 1
RIGHT = 2
BOTTOM = 3
DIRECTIONS = ["top", "left", "right", "bottom"]

ORDINALS = ['', "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

RAND_VOCAB = map(lambda x: "".join(x), list(itertools.product(string.ascii_lowercase,
    string.ascii_lowercase, string.ascii_lowercase, string.ascii_lowercase,
    string.ascii_lowercase))[:(TRAIN_SIZE_RANDOBJ+TEST_SIZE_RANDOBJ)])

CURRENT_VAR = 'a'
SPACE_VAR = 1
def getVar():
    global CURRENT_VAR
    var = CURRENT_VAR
    CURRENT_VAR = chr(ord(CURRENT_VAR)+1)
    return var

def getSpaceVar():
    global SPACE_VAR
    var = "w%d"%SPACE_VAR
    SPACE_VAR += 1
    return var

def resetVars():
    global CURRENT_VAR, SPACE_VAR
    CURRENT_VAR = 'a'
    SPACE_VAR = 1

class Shape:
    UPPER_LEFT_CHOICES = ["the upper left corner of %s", "%s's upper left corner"]
    UPPER_RIGHT_CHOICES = ["the upper right corner of %s", "%s's upper right corner"]
    LOWER_LEFT_CHOICES = ["the lower left corner of %s", "%s's lower left corner"]
    LOWER_RIGHT_CHOICES = ["the lower right corner of %s", "%s's lower right corner"]
    def __init__(self, name, width, height, description):
        self.name = name
        self.width = width
        self.height = height
        self.description = description
        self.logic = []
        self.left = 0
        self.right = width - 1
        self.top = 0
        self.bottom = height - 1
        self.var = getVar()
        self.ind = 0
        self.hasSize = True

    def getLowerLeftDescription(self):
        return random.choice(Shape.LOWER_LEFT_CHOICES)
    def getUpperLeftDescription(self):
        return random.choice(Shape.UPPER_LEFT_CHOICES)
    def getLowerRightDescription(self):
        return random.choice(Shape.LOWER_RIGHT_CHOICES)
    def getUpperRightDescription(self):
        return random.choice(Shape.UPPER_RIGHT_CHOICES)

    def getEnum(self, row, col):
        return row * self.width + col
    def getEdgeDescription(self, direction, offset, name):
        if direction == TOP or direction == BOTTOM:
            return "the " + ORDINALS[offset + 1] + " column of %s"%name
        else:
            return "the " + ORDINALS[offset + 1] + " row of %s"%name

    def fillIn(self, arr):
        for i in xrange(self.width):
            for j in xrange(self.height):
                arr[self.top + j][self.left+i] = 'X'

    def toString(self):
        print self.width
        print self.height

    def drawPrint(self):
        result = self.draw()
        for row in xrange(self.height-1, -1, -1):
            print " ".join(result[row])

class Relation(object):
    INLINE_CHOICES = ["is aligned with", "is in line with", "is next to", "is adjacent to"]
    NEXT_CHOICES = ["is next to", "is adjacent to"]
    def __init__(self, direction, second, first, offset, coarseIsPresent, fineIsPresent, isFlipped):
        self.direction = direction
        self.second = second
        self.first = first
        self.offset = offset
        self.description = "" 
        self.coarseIsPresent = coarseIsPresent
        self.fineIsPresent = fineIsPresent
        self.isFlipped = isFlipped

        
        if isFlipped:
            temp = first
            first = second
            second = temp
            if direction == TOP:
                direction = BOTTOM
            elif direction == BOTTOM:
                direction = TOP
            elif direction == LEFT:
                direction = RIGHT
            elif direction == RIGHT:
                direction = LEFT
            offset *= -1
        # Build the description
        # Recipe: [SECOND REFERENCE] [RELATIVE LOCATION] [FIRST REFERENCE]
        # note: for now, it is assumed that the second shape was just mentioned.
        if first.ind != 0:
            if isFlipped:
                firstName = "%s %s"%(random.choice(["this", "the new"]), first.name)
            else:
                firstName = "the %s %s"%(ORDINALS[first.ind], first.name)
        else:
            firstName = "the %s"%first.name
        if second.ind != 0:
            if isFlipped:
                secondName = "the %s %s"%(ORDINALS[second.ind], second.name)
            else:
                secondName = "%s %s"%(random.choice(["this", "the new"]), second.name)
        else:
            secondName = "the %s"%second.name
        if direction == LEFT:
            # SECOND REFERENCE
            if offset >= 0:
                self.description += second.getUpperRightDescription()%secondName + " "
                top = True
                middle = False
            elif second.bottom > first.bottom:
                offset *= -1
                self.description += second.getEdgeDescription(RIGHT, offset, secondName) + " "
                top = False
                middle = True
            else:
                self.description += second.getLowerRightDescription()%secondName + " "
                top = False
                middle = False

            # RELATIVE LOCATION
            if offset == 0 or second.bottom == first.bottom:
                self.description += random.choice(Relation.INLINE_CHOICES) + " "
            elif randint(0,1):
                self.description += random.choice(Relation.NEXT_CHOICES) + " "
            else:
                self.description += random.choice(["is to the left of", "is left of", "should be to the left of"]) + " "

            # FIRST REFERENCE
            if top:
                if second.top == first.top:
                    self.description += first.getUpperLeftDescription()%firstName
                elif second.top == first.bottom:
                    self.description += first.getLowerLeftDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(LEFT, offset, firstName)
            elif middle:
                self.description += first.getUpperLeftDescription()%firstName
            else:
                if second.bottom == first.top:
                    self.description += first.getUpperLeftDescription()%firstName
                elif second.bottom == first.bottom:
                    self.description += first.getLowerLeftDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(LEFT, offset+second.height-1, firstName)

        elif direction == RIGHT:
            # SECOND REFERENCE
            if offset >= 0:
                self.description += second.getUpperLeftDescription()%secondName + " "
                top = True
                middle = False
            elif second.bottom > first.bottom:
                offset *= -1
                self.description += second.getEdgeDescription(LEFT, offset, secondName) + " "
                top = False
                middle = True
            else:
                self.description += second.getLowerLeftDescription()%secondName + " "
                top = False
                middle = False

            # RELATIVE LOCATION
            if offset == 0:
                self.description += random.choice(Relation.INLINE_CHOICES) + " "
            elif random.randint(0,1):
                self.description += random.choice(Relation.NEXT_CHOICES) + " "
            else:
                self.description += random.choice(["is to the right of", "is right of", "should be to the right of"]) + " "

            # FIRST REFERENCE
            if top:
                if second.top == first.top:
                    self.description += first.getUpperRightDescription()%firstName
                elif second.top == first.bottom:
                    self.description += first.getLowerRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(RIGHT, offset, firstName)
            elif middle:
                self.description += first.getUpperRightDescription()%firstName
            else:
                if second.bottom == first.top:
                    self.description += first.getUpperRightDescription()%firstName
                elif second.bottom == first.bottom:
                    self.description += first.getLowerRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(RIGHT, offset+second.height-1, firstName)

        elif direction == TOP:
            # SECOND REFERENCE
            if offset >= 0:
                self.description += second.getLowerLeftDescription()%secondName + " "
                left = True
                middle = False
            elif second.right > first.right:
                offset *= -1
                self.description += second.getEdgeDescription(BOTTOM, offset, secondName) + " "
                left = False
                middle = True
            else:
                self.description += second.getLowerRightDescription()%secondName + " "
                left = False
                middle = False

            # RELATIVE LOCATION
            if offset == 0:
                self.description += random.choice(Relation.INLINE_CHOICES) + " "
            elif random.randint(0,1):
                self.description += random.choice(Relation.NEXT_CHOICES) + " "
            else:
                self.description += random.choice(["is above", "is to the top of", "should be above", "should be to the top of"]) + " "

            # FIRST REFERENCE
            if left:
                if second.left == first.left:
                    self.description += first.getUpperLeftDescription()%firstName
                elif second.left == first.right:
                    self.description += first.getUpperRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(TOP, offset, firstName)
            elif middle:
                self.description += first.getUpperLeftDescription()%firstName
            else:
                if second.right == first.left:
                    self.description += first.getUpperLeftDescription()%firstName
                elif second.right == first.right:
                    self.description += first.getUpperRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(TOP, offset + second.width -1, firstName)

        elif direction == BOTTOM:
            # SECOND REFERENCE
            if offset >= 0: 
                self.description += second.getUpperLeftDescription()%secondName + " "
                left = True
                middle = False
            elif second.right > first.right:
                offset *= -1
                self.description += second.getEdgeDescription(TOP, offset, secondName) + " "
                left = False
                middle = True
            else:
                self.description += second.getUpperRightDescription()%secondName + " "
                left = False
                middle = False

            # RELATIVE LOCATION
            if offset == 0:
                self.description += random.choice(Relation.INLINE_CHOICES) + " "
            elif random.randint(0,1):
                self.description += random.choice(Relation.NEXT_CHOICES) + " "
            else:
                self.description += random.choice(["is below", "is beneath", "is to the bottom of", "should be below", "should be beneath", "should be to the bottom of"]) + " "
            

            # FIRST REFERENCE
            if left:
                if second.left == first.left:
                    self.description += first.getLowerLeftDescription()%firstName
                elif second.left == first.right:
                    self.description += first.getLowerRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(TOP, offset, firstName)
            elif middle:
                self.description += first.getLowerLeftDescription()%firstName
            else:
                if second.right == first.left:
                    self.description += first.getLowerLeftDescription()%firstName
                elif second.right == first.right:
                    self.description += first.getLowerRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(TOP, offset + second.width-1, firstName)

#TODO: Make sure offsets are correct!
class CompositeShape(object):
    def __init__(self):
        self.shapes = None
        self.shapesOnSides = [None]*4
        self.totalHeight = 0
        self.totalWidth = 0
        self.relations = []
        self.logic = []
        self.description = None

    def addShape(self, shape, direction, coarseIsPresent, fineIsPresent):
        if not self.shapes:
            self.shapes = [shape]
            #The following keeps track of which shape is on each side.
            self.shapesOnSides = [shape]*4
            self.maxX = shape.right
            self.minX = 0
            self.maxY = shape.bottom
            self.minY = 0
            if shape.hasSize:
                self.logic.append(shape.getLogic())
            else:
                self.logic.append(shape.getCondensedLogic())
        else:
            finalInd = 1
            for oldShape in self.shapes:
                if oldShape.name == shape.name: 
                    if oldShape.ind == 0:
                        oldShape.ind = 1
                        finalInd = 2
                        break
                    else:
                        finalInd += 1
            if finalInd > 1:
                shape.ind = finalInd

            self.shapes.append(shape)
            newLogic = []
            self.logic.append(newLogic)
            if shape.hasSize:
                newLogic += shape.getLogic()
            else:
                newLogic += shape.getCondensedLogic() 
            def getAlignmentPred(shape, pred, num=None):
                if num != None:
                    if shape.base_name == "row":
                        return "block-col-ind(%s, %s, "+num+")"
                    elif shape.base_name == "col":
                        return "block-row-ind(%s, %s, "+num+")"
                    else:
                        return pred
                else:
                    if shape.base_name == "row":
                        if "right" in pred:
                            return "right-end(%s, %s)"
                        else:
                            return "left-end(%s, %s)"
                    elif shape.base_name == "col":
                        if "upper" in pred:
                            return "top-end(%s, %s)"
                        else:
                            return "bottom-end(%s, %s)"
                    else:
                        return pred

            isFlipped = random.randint(0,1)


            if direction == LEFT:
                old = self.shapesOnSides[LEFT]
                #newLogic += old.getCondensedLogic()
                offset = randint(-1*(shape.height - 1), old.height - 1)
                shape.right = old.left - 1
                shape.left = old.left - shape.width
                shape.top = old.top + offset
                shape.bottom = old.top + offset + shape.height - 1
                
                self.relations.append(Relation(LEFT, shape, old, offset, coarseIsPresent, fineIsPresent, isFlipped))
                self.shapesOnSides[LEFT] = shape
                if shape.left <= self.minX:
                    self.minX = shape.left
                    self.shapesOnSides[LEFT] = shape
                if shape.top <= self.minY:
                    self.minY = shape.top
                    self.shapesOnSides[TOP] = shape
                if shape.bottom >= self.maxY:
                    self.maxY = shape.bottom
                    self.shapesOnSides[BOTTOM] = shape
                if coarseIsPresent:
                    newLogic.append("left(%s, %s)"%(shape.var, old.var))
                
            elif direction == RIGHT:
                old = self.shapesOnSides[RIGHT]
                #newLogic += old.getCondensedLogic()
                offset = randint(-1*(shape.height - 1), old.height - 1)
                shape.left = old.right + 1
                shape.right = old.right + shape.width
                shape.top = old.top + offset
                shape.bottom = old.top + offset + shape.height - 1
                self.relations.append(Relation(RIGHT, shape, old, offset, coarseIsPresent, fineIsPresent, isFlipped))
                if shape.right >= self.maxX:
                    self.maxX = shape.right
                    self.shapesOnSides[RIGHT] = shape
                if shape.top <= self.minY:
                    self.minY = shape.top
                    self.shapesOnSides[TOP] = shape
                if shape.bottom >= self.maxY:
                    self.maxY = shape.bottom
                    self.shapesOnSides[BOTTOM] = shape
                if coarseIsPresent:
                    newLogic.append("right(%s, %s)"%(shape.var, old.var))

            elif direction == TOP:
                old = self.shapesOnSides[TOP]
                offset = randint(-1*(shape.width-1), old.width - 1)
                shape.bottom = old.top - 1
                shape.top = old.top - shape.height
                shape.left = old.left + offset
                shape.right = old.left + offset + shape.width - 1
                self.relations.append(Relation(TOP, shape, old, offset, coarseIsPresent, fineIsPresent, isFlipped))
                if shape.left <= self.minX:
                    self.minX = shape.left
                    self.shapesOnSides[LEFT] = shape
                if shape.right >= self.maxX:
                    self.maxX = shape.right
                    self.shapesOnSides[RIGHT] = shape
                if shape.top <= self.minY:
                    self.minY = shape.top
                    self.shapesOnSides[TOP] = shape
                if coarseIsPresent:
                    newLogic.append("top(%s, %s)"%(shape.var, old.var))

            elif direction == BOTTOM:
                old = self.shapesOnSides[BOTTOM]
                offset = randint(-1*(shape.width-1), old.width - 1)
                shape.top = old.bottom + 1
                shape.bottom = old.bottom + shape.height
                shape.left = old.left + offset
                shape.right = old.left + offset + shape.width - 1
                self.relations.append(Relation(BOTTOM, shape, old, offset, coarseIsPresent, fineIsPresent, isFlipped))
                if shape.left <= self.minX:
                    self.minX = shape.left
                    self.shapesOnSides[LEFT] = shape
                if shape.right >= self.maxX:
                    self.maxX = shape.right
                    self.shapesOnSides[RIGHT] = shape
                if shape.bottom >= self.maxY:
                    self.maxY = shape.bottom
                    self.shapesOnSides[BOTTOM] = shape

                if coarseIsPresent:
                    newLogic.append("bottom(%s, %s)"%(shape.var, old.var))

                
            if isFlipped:
                temp = shape
                shape = old
                old = temp
                if direction == TOP:
                    direction = BOTTOM
                elif direction == BOTTOM:
                    direction = TOP
                elif direction == LEFT:
                    direction = RIGHT
                elif direction == RIGHT:
                    direction = LEFT
                offset *= -1

            if direction == LEFT:
                # Additional Logic
                if offset >= 0:
                    newRow = 0
                    newCol = shape.width-1
                    newEnumLogic = getAlignmentPred(shape, "upper_right(%s, %s)")
                    oldRow = offset
                    oldCol = 0
                    if offset == 0:
                        oldEnumLogic = getAlignmentPred(old, "upper_left(%s, %s)")
                    elif shape.top == old.bottom:
                        oldEnumLogic = getAlignmentPred(old, "lower_left(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "left_side(%s, %s, "+str(offset)+")", str(offset))
                elif shape.bottom > old.bottom:
                    #In this case, use the corner of the old shape
                    oldEnumVal = old.getEnum(0, 0)
                    oldRow = 0
                    oldCol = 0
                    oldEnumLogic = getAlignmentPred(old, "upper_left(%s, %s)")
                    newEnumVal = shape.getEnum(-1*offset, shape.width-1)
                    newRow = -1*offset
                    newCol = shape.width-1
                    newEnumLogic = getAlignmentPred(shape, "right_side(%s, %s, "+str(-1*offset)+")", str(-1*offset))
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, shape.width - 1)
                    newRow = shape.height-1
                    newCol = shape.width-1
                    newEnumLogic = getAlignmentPred(shape, "lower_right(%s, %s)")
                    oldEnumVal = old.getEnum(shape.bottom - old.top, 0)
                    oldRow = shape.bottom - old.top
                    oldCol = 0
                    if shape.bottom == old.bottom:

                        oldEnumLogic = getAlignmentPred(old, "lower_left(%s, %s)")
                    elif shape.bottom == old.top:
                        oldEnumLogic = getAlignmentPred(old, "upper_left(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "left_side(%s, %s, "+str(shape.bottom-old.top) + ")", str(shape.bottom-old.top))
                if fineIsPresent:
                    newLogic = []
                    self.logic.append(newLogic)
                    newSpaceVar = getSpaceVar()
                    oldSpaceVar = getSpaceVar()
                    newBlockVar = getVar()
                    oldBlockVar = getVar()
                    newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [oldEnumLogic%(old.var, newBlockVar)]
                    #else:
                    newLogic += [newEnumLogic%(shape.var, newBlockVar)]
                    #newLogic += ["row-ind(%s, %s, %d)"%(shape.var, newBlockVar, newRow), "col-ind(%s, %s, %d)"%(shape.var, newBlockVar, newCol)]
                    newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [newEnumLogic%(shape.var,oldBlockVar)]
                    #    newLogic.append("spatial-rel(east, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))
                    #else:
                    newLogic += [oldEnumLogic%(old.var,oldBlockVar)]
                    newLogic.append("spatial-rel(west, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))

            
            elif direction == RIGHT:
                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, 0)
                    newRow = 0
                    newCol = 0
                    newEnumLogic = getAlignmentPred(shape, "upper_left(%s, %s)")
                    oldEnumVal = old.getEnum(offset, old.width-1)
                    oldRow = offset
                    oldCol = old.width-1
                    if offset == 0:
                        oldEnumLogic = getAlignmentPred(old, "upper_right(%s, %s)")
                    elif shape.top == old.bottom:
                        oldEnumLogic = getAlignmentPred(old, "lower_right(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "right_side(%s, %s, "+str(offset) +")", str(offset))
                elif shape.bottom > old.bottom:
                    #In this case, use the corner of the old shape
                    oldEnumVal = old.getEnum(0, old.width-1)
                    oldRow = 0
                    oldCol = old.width-1
                    oldEnumLogic = getAlignmentPred(old, "upper_right(%s, %s)")
                    newEnumVal = shape.getEnum(-1*offset, 0)
                    newRow = -1*offset
                    newCol = 0
                    newEnumLogic = getAlignmentPred(shape, "left_side(%s, %s, "+str(-1*offset)+")", str(-1*offset))
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, 0)
                    newRow = shape.height-1
                    newCol = 0
                    newEnumLogic = getAlignmentPred(shape, "lower_left(%s, %s)")
                    oldEnumVal = old.getEnum(shape.bottom - old.top, old.width-1)
                    oldRow = shape.bottom - old.top
                    oldCol = old.width-1
                    if shape.bottom == old.bottom:
                        oldEnumLogic = getAlignmentPred(old, "lower_right(%s, %s)")
                    elif shape.bottom == old.top:
                        oldEnumLogic = getAlignmentPred(old, "upper_right(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "right_side(%s, %s, "+str(shape.bottom-old.top)+")", str(shape.bottom-old.top))
                if fineIsPresent:
                    newSpaceVar = getSpaceVar()
                    oldSpaceVar = getSpaceVar()
                    newBlockVar = getVar()
                    oldBlockVar = getVar()
                    newLogic = []
                    self.logic.append(newLogic)
                    newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [oldEnumLogic%(old.var, newBlockVar)]
                    #else:
                    newLogic += [newEnumLogic%(shape.var, newBlockVar)]
                    #newLogic += ["row-ind(%s, %s, %d)"%(shape.var, newBlockVar, newRow), "col-ind(%s, %s, %d)"%(shape.var, newBlockVar, newCol)]
                    newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [newEnumLogic%(shape.var, oldBlockVar)]
                    #    newLogic.append("spatial-rel(west, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))
                    #else:
                    newLogic += [oldEnumLogic%(old.var,oldBlockVar)]
                    newLogic.append("spatial-rel(east, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))
            
            elif direction == TOP:
                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(shape.height - 1, 0)
                    newRow = shape.height-1
                    newCol = 0
                    newEnumLogic = getAlignmentPred(shape, "lower_left(%s, %s)")
                    oldEnumVal = old.getEnum(0, offset)
                    oldRow = 0
                    oldCol = offset
                    if offset == 0:
                        oldEnumLogic = getAlignmentPred(old, "upper_left(%s, %s)")
                    elif shape.left == old.right:
                        oldEnumLogic = getAlignmentPred(old, "upper_right(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "top_side(%s, %s, "+str(offset)+")", str(offset))
                elif shape.right > old.right:
                    oldEnumVal = old.getEnum(0, 0)
                    oldRow = 0
                    oldCol = 0
                    oldEnumLogic = getAlignmentPred(old, "upper_left(%s, %s)")
                    newEnumVal = shape.getEnum(shape.height-1, -1*offset)
                    newRow = shape.height-1
                    newCol = -1*offset
                    newEnumLogic = getAlignmentPred(shape, "bottom_side(%s, %s, "+str(-1*offset)+")", str(-1*offset))
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, shape.width - 1)
                    newRow = shape.height-1
                    newCol = shape.width-1
                    newEnumLogic = getAlignmentPred(shape, "lower_right(%s, %s)")
                    oldEnumVal = old.getEnum(0, shape.right - old.left)
                    oldRow = 0
                    oldCol = shape.right-old.left
                    if old.right == shape.right:
                        oldEnumLogic = getAlignmentPred(old, "upper_right(%s, %s)")
                    elif shape.right == old.left:
                        oldEnumLogic = getAlignmentPred(old, "upper_left(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "top_side(%s, %s, "+str(shape.right - old.left)+")", str(shape.right - old.left))
                if fineIsPresent:
                    newLogic = []
                    self.logic.append(newLogic)
                    newSpaceVar = getSpaceVar()
                    oldSpaceVar = getSpaceVar()
                    newBlockVar = getVar()
                    oldBlockVar = getVar()
                    newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                    #if isFlipped:
                    #newLogic += [oldEnumLogic%(old.var, newBlockVar)]
                    #else:
                    newLogic += [newEnumLogic%(shape.var, newBlockVar)]
                    newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [newEnumLogic%(shape.var,oldBlockVar)]
                    #    newLogic.append("spatial-rel(south, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))
                    #else:
                    newLogic += [oldEnumLogic%(old.var,oldBlockVar)]
                    newLogic.append("spatial-rel(north, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))
            
            elif direction == BOTTOM:
                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, 0)
                    newRow = 0
                    newCol = 0
                    newEnumLogic = getAlignmentPred(shape, "upper_left(%s, %s)")
                    oldEnumVal = old.getEnum(old.height - 1, offset)
                    oldRow = old.height-1
                    oldCol = offset
                    if offset == 0:
                        oldEnumLogic = getAlignmentPred(old, "lower_left(%s, %s)")
                    elif shape.left == old.right:
                        oldEnumLogic = getAlignmentPred(old, "lower_right(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "bottom_side(%s, %s, "+str(offset)+")", str(offset))
                elif shape.right > old.right:
                    oldEnumVal = old.getEnum(old.height-1, 0)
                    oldRow = old.height-1
                    oldCol = 0
                    oldEnumLogic = getAlignmentPred(old, "lower_left(%s, %s)")
                    newEnumVal = shape.getEnum(0, -1*offset)
                    newRow = 0
                    newCol = -1*offset
                    newEnumLogic = getAlignmentPred(shape, "top_side(%s, %s, "+str(-1*offset)+")", str(-1*offset))
                else:
                    newEnumVal = shape.getEnum(0, shape.width - 1)
                    newRow = 0
                    newCol = shape.width-1
                    newEnumLogic = getAlignmentPred(shape, "upper_right(%s, %s)")
                    oldEnumVal = old.getEnum(old.height - 1, shape.right - old.left)
                    oldRow = old.height-1
                    oldCol = shape.right-old.left
                    if old.right == shape.right:
                        oldEnumLogic = getAlignmentPred(old, "lower_right(%s, %s)")
                    elif shape.right == old.left:
                        oldEnumLogic = getAlignmentPred(old, "lower_left(%s, %s)")
                    else:
                        oldEnumLogic = getAlignmentPred(old, "bottom_side(%s, %s, "+str(shape.right - old.left)+")", str(shape.right - old.left))
                if fineIsPresent:
                    newLogic = []
                    self.logic.append(newLogic)
                    newSpaceVar = getSpaceVar()
                    oldSpaceVar = getSpaceVar()
                    newBlockVar = getVar()
                    oldBlockVar = getVar()
                    newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [oldEnumLogic%(old.var, newBlockVar)]
                    #else:
                    newLogic += [newEnumLogic%(shape.var, newBlockVar)]
                    newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                    newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                    #if isFlipped:
                    #    newLogic += [newEnumLogic%(shape.var, oldBlockVar)]
                    #    newLogic.append("spatial-rel(north, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))
                    #else:
                    newLogic += [oldEnumLogic%(old.var,oldBlockVar)]
                    newLogic.append("spatial-rel(south, 0, %s, %s)"%(newSpaceVar, oldSpaceVar))

    def addRandomConstraint(self):
        num = randint(1,3)
        restricted = []
        new_formula = []
        for i in xrange(num):
            xConst = randint(0,10)
            yConst = randint(0,10)
            while [xConst, yConst] in restricted:
                xConst = randint(0,10)
                yConst = randint(0,10)
            restricted.append([xConst, yConst])
            new_formula.append("restricted(%d, %d)"%(xConst, yConst))
        self.getDescription()
        self.logic.append(new_formula)
        verb = random.choice(["put", "place"])
        if num == 1:
            self.description.append("Do not %s a block in the space at row %d and column %d."%(verb, restricted[0][1], restricted[0][0]))
        elif num == 2:
            self.description.append("Do not %s blocks in the spaces at row %d and column %d and at row %d and column %d."%(verb, restricted[0][1], restricted[0][0], restricted[1][1], restricted[1][0]))
        else:
            result = "Do not %s blocks in "%verb
            for i in xrange(num-1):
                result += "the space at row %d and column %d, "%(restricted[i][1], restricted[i][0])
            result += "and the space at row %d and column %d."%(restricted[-1][1], restricted[-1][0])
            self.description.append(result)
        self.clarification.append([])


    def addStartBlocks(self):
        num = randint(1,3)
        restricted = []
        new_formula = []
        for i in xrange(num):
            xConst = randint(0,10)
            yConst = randint(0,10)
            while [xConst, yConst] in restricted:
                xConst = randint(0,10)
                yConst = randint(0,10)
            restricted.append([xConst, yConst])
            new_formula.append("starting_block(%d, %d)"%(xConst, yConst))
        self.logic.append(new_formula)
        self.getDescription()
        if num == 1:
            self.description.append("There is already a block at row %d and column %d."%(restricted[0][1], restricted[0][0]))
        elif num == 2:
            self.description.append("There are already blocks at row %d and column %d and at row %d and column %d."%(restricted[0][1], restricted[0][0], restricted[1][1], restricted[1][0]))
        else:
            result = "There are already blocks "
            for i in xrange(num-1):
                result += "at row %d and column %d, "%(restricted[i][1], restricted[i][0])
            result += "and at row %d and column %d."%(restricted[-1][1], restricted[-1][0])
            self.description.append(result)
        self.clarification.append([])

    def addImmobileBlocks(self):
        num = randint(1,3)
        restricted = []
        new_formula = []
        for i in xrange(num):
            xConst = randint(0,9)
            yConst = randint(0,9)
            while [xConst, yConst] in restricted:
                xConst = randint(0,9)
                yConst = randint(0,9)
            restricted.append([xConst, yConst])
            new_formula.append("immobile_block(%d, %d)"%(xConst, yConst))
        self.logic.append(new_formula)
        self.getDescription()
        imm = random.choice(["immovable", "immobile"])
        if num == 1:
            self.description.append("There is an %s block at row %d and column %d."%(imm, restricted[0][1], restricted[0][0]))
        elif num == 2:
            self.description.append("There are %s blocks at row %d and column %d and at row %d and column %d."%(imm, restricted[0][1], restricted[0][0], restricted[1][1], restricted[1][0]))
        else:
            result = "There are %s blocks "%imm
            for i in xrange(num-1):
                result += "at row %d and column %d, "%(restricted[i][1], restricted[i][0])
            result += "and at row %d and column %d."%(restricted[-1][1], restricted[-1][0])
            self.description.append(result)
        self.clarification.append([])

    def normalize(self):
        xShift = -1 * self.minX
        self.minX += xShift
        self.maxX += xShift
        yShift = -1 * self.minY
        self.minY += yShift
        self.maxY += yShift

        if self.maxX >= GRID_WIDTH or self.maxY >= GRID_HEIGHT:
            return False
        #if self.maxX < GRID_WIDTH - 1:
        #    xShift += (GRID_WIDTH - self.maxX - 1) / 2
        #if self.maxY < GRID_HEIGHT - 1:
        #    yShift += (GRID_HEIGHT - self.maxY - 1) / 2

        for shape in self.shapes:
            shape.left += xShift
            shape.right += xShift
            shape.top += yShift
            shape.bottom += yShift
        return True

    def getDescription(self):
        if self.description:
            return self.description, self.clarification

        self.description = []
        self.clarification = []
        if len(self.shapes) == 1:
            description = random.choice(COMMANDS) + " "
            if not self.shapes[0].hasSize:
                self.clarification.append([description + self.shapes[0].getDescription(True)+"."])
            else:
                self.clarification.append([])
            description += self.shapes[0].getDescription() + "."
            self.description.append([description])
        elif len(self.shapes) >= 2:
            description = random.choice(COMMANDS) + " "
            if not self.shapes[0].hasSize:
                self.clarification.append([description + self.shapes[0].getDescription(True)+"."])
            else:
                self.clarification.append([])
            description += self.shapes[0].getDescription() + "."
            self.description.append([description])
            for i in xrange(1, len(self.shapes)):
                shape_description = []
                self.description.append(shape_description)
                description = ""
                clarification = []
                self.clarification.append(clarification)
                if random.randint(0,1):
                    if i == len(self.shapes) - 1 and random.randint(0,1):
                        #description += "Finally, %s "%random.choice(COMMANDS).lower()
                        start = "Finally, "
                    else:
                        start = "%s, "%random.choice(NEXT)
                        #description += "%s "%random.choice(COMMANDS).lower()
                else:
                    start = ""
                command = random.choice(COMMANDS)
                '''
                if self.relations[i-1].first.base_name == self.shapes[i].base_name and random.randint(0,1):
                    description += "another %s "%self.shapes[i].name
                    if random.randint(0,1) == 1 and self.relations[i-1].direction in [TOP, BOTTOM]:
                        if self.relations[i-1].direction == TOP:
                            description += "above the first one."
                        else:
                            description += "%s the first one."%random.choice(["below", "beneath"])
                    else:
                        description += "to the %s of the first one." %(DIRECTIONS[self.relations[i-1].direction])
                    #description += self.shapes[1].size_description + " to the %s of the first one." %(DIRECTIONS[self.relations[0].direction])
                    clarification.append(None)
                
                else:
                '''
                name = "the %s"%self.relations[i-1].first.name
                if not self.shapes[i].hasSize:
                    needsClarification = True
                else:
                    needsClarification = False
            
                if random.randint(0,1) == 1 and self.relations[i-1].direction in [TOP, BOTTOM]:
                    if self.relations[i-1].direction == TOP:
                        addon = "above %s"%(name)
                    else:
                        addon =  "%s %s"%(random.choice(["below", "beneath"]),name)
                else:
                    addon = "to the %s of %s" %(DIRECTIONS[self.relations[i-1].direction], name)
                if random.randint(0,1):
                    if start != "":
                        command = command[0].lower() + command[1:]
                        start = start[0].upper() + start[1:]
                    description = start + command + " "+self.shapes[i].getDescription()
                    if self.relations[i-1].coarseIsPresent:
                        description += " " + addon
                        needsClarification = True
                    description += "."
                    if needsClarification:
                        clarification.append(start + command + " "+self.shapes[i].getDescription(True) + " " + addon + ".")
                    else:
                        clarification.append(None)
                else:
                    if self.relations[i-1].coarseIsPresent:
                        command = command[0].lower() + command[1:]
                        if start == "":
                            addon = addon[0].upper() + addon[1:]
                        else:
                            start = start[0].upper() + start[1:]
                        description = start + addon + ", "+ command + " " + self.shapes[i].getDescription() + "."
                    else:
                        if start != "":
                            start = start[0].upper() + start[1:]
                            command = command[0].lower() + command[1:]
                        description = start + command + " " + self.shapes[i].getDescription() + "."
                        needsClarification=True
                    if needsClarification:
                        clarification.append(start + addon + ", " + command + " " + self.shapes[i].getDescription(True) + ".")
                    else:
                        clarification.append(None)
                shape_description.append(description)
                option = random.randint(0,3)
                if option == 0:
                    relation = "Ensure that %s."%self.relations[i-1].description
                elif option == 1:
                    relation = "Make sure that %s."%self.relations[i-1].description
                elif option == 2:
                    relation = "Check that %s."%self.relations[i-1].description
                elif option == 3:
                    relation = self.relations[i-1].description + "."
                    relation = relation[0].upper()+relation[1:]
                if self.relations[i-1].fineIsPresent:
                    shape_description.append(relation)
                else:
                    clarification.append(relation)
                #shape_description.append(description)

        return self.description, self.clarification


    def draw(self):
        result = [['O']*(GRID_WIDTH) for i in xrange(GRID_HEIGHT)]
        for shape in self.shapes:
            shape.fillIn(result)
        return result

    def write_experiment(self, fout):
        result = self.draw()
        for row in xrange(len(result)):
            fout.write(" ".join(result[row]))
            fout.write("\n")
        fout.write("===\n")
        descriptions, clarifications = self.getDescription()
        for i,description in enumerate(descriptions):
            #fout.write(' ^ '.join(self.logic[i]))
            #fout.write("\n")
            fout.write(" ".join(description))
            for entry in clarifications[i]:
                if entry != None:
                    fout.write("\n")
                    fout.write("C:"+entry)
            fout.write("\n")
        fout.write("#")

    def write_train(self, fout):
        descriptions, _ = self.getDescription()
        print descriptions
        print self.logic
        print ""
        modified_descriptions = [item for sublist in descriptions for item in sublist]
        for description,logic in zip(modified_descriptions, self.logic):
            fout.write(' ^ '.join(logic))
            fout.write("\n")
            fout.write(description)
            fout.write("\n")

    def draw_to_file(self, file_path):
        shape = self.draw()


class Row(Shape):
    DESCRIPTIONS = ["row", "horizontal line"]
    COMPOSITIONS = ["using %d blocks", "of size %d", "of length %d", "with %d blocks"]
    LEFT_CHOICES = ["the left end of %s", "%s's left end"]
    RIGHT_CHOICES = ["the right end of %s", "%s's right end"]
    def __init__(self, length):
        self.base_name = "row"
        name = random.choice(Row.DESCRIPTIONS)
        self.size_description = random.choice(Row.COMPOSITIONS)%length
        description = "a %s %s"%(name, self.size_description)
        #                   name, width, height, description
        Shape.__init__(self, name, length, 1, description)

    def getLowerLeftDescription(self):
        return random.choice(Row.LEFT_CHOICES)
    def getUpperLeftDescription(self):
        return random.choice(Row.LEFT_CHOICES)
    def getLowerRightDescription(self):
        return random.choice(Row.RIGHT_CHOICES)
    def getUpperRightDescription(self):
        return random.choice(Row.RIGHT_CHOICES)

    def getLogic(self):
        return ['row(%s)'%self.var, 'width(%s, %d)'%(self.var, self.width)]

    def getCondensedLogic(self):
        return ['row(%s)'%self.var]
    def getDescription(self, override=False):
        if self.hasSize or override:
            return self.description
        else:
            return "a %s"%self.name


class Col(Shape):
    DESCRIPTIONS = ["column", "vertical line"]
    COMPOSITIONS = ["using %d blocks", "of size %d", "of height %d", "with %d blocks"]
    TOP_CHOICES = ["the top end of %s", "%s's top end"]
    BOTTOM_CHOICES = ["the bottom end of %s", "%s's bottom end"]
    def __init__(self, height):
        self.base_name = "col"
        name = random.choice(Col.DESCRIPTIONS)
        self.size_description = random.choice(Col.COMPOSITIONS)%height
        description = "a %s %s"%(name, self.size_description)
        #                   name, width, height, description
        Shape.__init__(self, name, 1, height, description)

    def getLowerLeftDescription(self):
        return random.choice(Col.BOTTOM_CHOICES)
    def getUpperLeftDescription(self):
        return random.choice(Col.TOP_CHOICES)
    def getLowerRightDescription(self):
        return random.choice(Col.BOTTOM_CHOICES)
    def getUpperRightDescription(self):
        return random.choice(Col.TOP_CHOICES)

    def getLogic(self):
        return ['column(%s)'%self.var, 'height(%s, %d)'%(self.var, self.height)]
    def getCondensedLogic(self):
        return ['column(%s)'%self.var]
    def getDescription(self, override=False):
        if self.hasSize or override:
            return self.description
        else:
            return "a %s"%self.name

class Square(Shape):
    def __init__(self, dim):
        self.base_name = "square"
        DESCRIPTIONS = ["a %d by %d square"%(dim, dim),
            "a square with sides of block length %d"%dim,
            "a %d block by %d block square"%(dim, dim)]
        self.size_description = "with sides of block length %d"%dim
        description = random.choice(DESCRIPTIONS)
        #                   name, width, height, description
        Shape.__init__(self, "square", dim, dim, description)

    def getLogic(self):
        return ['square(%s)'%self.var, 'size(%s, %d)'%(self.var, self.width)]
    def getCondensedLogic(self):
        return ['square(%s)'%self.var]
    def getDescription(self, override=False):
        if self.hasSize or override:
            return self.description
        else:
            return "a square"

class Rect(Shape):
    def __init__(self, width, height):
        DESCRIPTIONS = ["a %d by %d rectangle"%(width, height), 
                        "a rectangle with a width of %d blocks and a height of %d blocks"%(width, height),
                        "a rectangle with a height of %d blocks and a width of %d blocks"%(height, width),
                        "a rectangle that is %d blocks wide and %d blocks tall"%(width, height),
                        "a rectangle that is %d blocks tall and %d blocks wide"%(height, width)]
        self.base_name = "rectangle"
        self.size_description = random.choice(DESCRIPTIONS[1:])[12:]
        description = random.choice(DESCRIPTIONS)
        #                   name, width, height, description,
        Shape.__init__(self, "rectangle", width, height, description)

    def getLogic(self):
        return ['rectangle(%s)'%self.var, 'height(%s, %d)'%(self.var, self.height), 'width(%s, %d)'%(self.var, self.width)]
    def getCondensedLogic(self):
        return ['rectangle(%s)'%self.var]
    def getDescription(self, override=False):
        if self.hasSize or override:
            return self.description
        else:
            return "a rectangle"

def randRow():
    length = randint(2,10)
    return Row(length)

def randCol():
    height = randint(2,10)
    return Col(height)

def randSquare():
    dim = randint(1,10)
    return Square(dim)

def randRect():
    length = randint(1,10)
    height = randint(1,10)
    return Rect(length, height)

genShape = {ROW:randRow, COL:randCol, SQUARE:randSquare, RECT:randRect}

def randomObjDescription(objName):
    options = [
        "Construct %s.",
        "Build %s.",
        "Place %s on the table.",
        "Put %s on the table.",
        "Create %s.",
        "You should construct %s.",
        "You should build %s.",
        "You should create %s.",
        "You should place %s on the table.",
        "You should put %s on the table.",
        "I want you to construct %s.",
        "I want you to build %s.",
        "I want you to create %s.",
        "I want you to place %s on the table.",
        "Please construct %s.",
        "Please build %s.",
        "Please place %s on the table.",
        "Please put %s on the table.",
        "Please create %s.",
    ]
    return random.choice(options)%objName

def genConfig(numShapes, numMissing):
    resetVars()
    randomVec = [True]*(3*(numShapes-1)+1)
    if IS_TRAIN_MODE:
        inds = np.random.choice(range(3*(numShapes-1)+1),size=numMissing, replace=False)
    else:
        inds = np.random.choice(range(2*(numShapes-1)+1),size=numMissing, replace=False)
        map(lambda x: x + x//2, inds)
    for ind in inds:
        randomVec[ind] = False
    composite = CompositeShape()
    prevShapeNum = None
    for i in xrange(numShapes):
        newShapeNum = randint(0,3)
        '''
        if newShapeNum == prevShapeNum:
            if newShapeNum == ROW:
                newShape = Row(prevShape.width)
            elif newShapeNum == COL:
                newShape = Col(prevShape.height)
            elif newShapeNum == SQUARE:
                newShape = Square(prevShape.width)
            else:
                newShape = Rect(prevShape.width, prevShape.height)
        else:
        '''
        newShape = genShape[newShapeNum]()
        direction = randint(0,3)
        if i == 0:
            newShape.hasSize = randomVec[i]
            composite.addShape(newShape, direction, None, None)
        else:
            newShape.hasSize = randomVec[3*(i-1)+1]
            composite.addShape(newShape, direction, randomVec[3*i-1],randomVec[3*i])
        prevShape = newShape
        prevShapeNum = newShapeNum
    '''
    constr = randint(0,3)
    if constr == 1:
        composite.addRandomConstraint()
    elif constr == 2:
        composite.addStartBlocks()
    elif constr == 3:
        composite.addImmobileBlocks()
    '''
    return composite


configs = []
descriptions = []
if len(sys.argv) != 3:
    print "Usage: python generate.py <train output name> <test output name>"
    sys.exit(1)

if IS_TRAIN_MODE:
    shapes = []
    descriptions = set()
    while len(shapes) < TRAIN_SIZE_SHAPES + TEST_SIZE_SHAPES:
        numShapes = randint(1, maxNumShapes)
        if randint(0,1):
            numMissing = 0
        else:
            numMissing = randint(1,3*(numShapes - 1)+1)
        composite = genConfig(numShapes, numMissing)
        description, clarification = composite.getDescription()
        flat_description = [item for sublist in description for item in sublist]
        if ''.join(flat_description) not in descriptions and composite.normalize():
            descriptions.add(''.join(flat_description))
            shapes.append(composite)

    randobjs = []
    for i in xrange(TRAIN_SIZE_RANDOBJ + TEST_SIZE_RANDOBJ):
        option = random.randint(0,3)
        article = random.choice(["a", "an"])
        letter = random.choice(string.ascii_uppercase)
        if option == 0:
            word = "%s"%(RAND_VOCAB[i])
            logic_form = "%s(a)"%RAND_VOCAB[i]
        else:
            logic_form = "%s(a)"%letter
            if option == 1:
                word = "letter %s"%letter
                article = "the"
            elif option == 2:
                word = "letter %s"%letter
            elif option == 3:
                word = letter
        option = randint(0,3)
        if option == 0:
            width = randint(2,10)
            logic_form += " ^ width(a, %d)"%width
            description = randomObjDescription("%s %s of %s %d"%(article, word, random.choice(['length', 'width']),width))
        elif option == 1:
            height = randint(2,10)
            logic_form += " ^ height(a, %d)"%height
            description = randomObjDescription("%s %s of height %d"%(article, word, height))
        elif option == 2:
            width = randint(2,10)
            height = randint(2,10)
            option = random.randint(0,3)
            if option == 0:
                logic_form += " ^ height(a, %d) ^ width(a, %d)"%(height, width)
                description = randomObjDescription("%s %s of height %d and %s %d"%(article, word, height, random.choice(['length', 'width']),width))
            elif option == 1:
                logic_form += " ^ width(a, %d) ^ height(a, %d)"%(width, height)
                description = randomObjDescription("%s %s of %s %d and height %d"%(article, word, random.choice(['length', 'width']), width, height))
            elif option == 2:
                logic_form += " ^ width(a, %d) ^ height(a, %d)"%(width, height)
                description = randomObjDescription("%s %d by %d %s"%(article, width, height, word))
            elif option == 3:
                logic_form += " ^ width(a, %d) ^ height(a, %d)"%(width, height)
                description = randomObjDescription("%s %s of size %d by %d"%(article, word, width, height))

                
        else:
            description = randomObjDescription("%s %s"%(article, word))
        randobjs.append((logic_form, description))
    
    train_data = shapes[:TRAIN_SIZE_SHAPES] + randobjs[:TRAIN_SIZE_RANDOBJ]
    random.shuffle(train_data)
    test_data = shapes[TRAIN_SIZE_SHAPES:] + randobjs[TRAIN_SIZE_RANDOBJ:]
    random.shuffle(test_data)

    with open(sys.argv[1], "w") as fout:
        for item in train_data:
            if isinstance(item, tuple):
                fout.write("%s\n%s\n"%item)
            else:
                item.write_train(fout)
            fout.write("\n")
    with open(sys.argv[2], "w") as fout:
        for item in test_data:
            if isinstance(item, tuple):
                fout.write("%s\n%s\n"%item)
            else:
                item.write_train(fout)
            fout.write("\n")
    
else:
    shapes = []
    descriptions = []
    for i in xrange(25):
        passed = False
        while not passed:
            composite = genConfig(1, 0)
            description, clarification = composite.getDescription()
            if ''.join(description) not in descriptions and composite.normalize():
                descriptions.add(''.join(description))
                shapes.append(composite)
                passed = True
    for i in xrange(25):
        passed = False
        while not passed:
            composite = genConfig(2, 0)
            description, clarification = composite.getDescription()
            if ''.join(description) not in descriptions and composite.normalize():
                descriptions.add(''.join(description))
                shapes.append(composite)
                passed = True
    for i in xrange(25):
        passed = False
        while not passed:
            composite = genConfig(3, 0)
            description, clarification = composite.getDescription()
            if ''.join(description) not in descriptions and composite.normalize():
                descriptions.add(''.join(description))
                shapes.append(composite)
                passed = True
    for i in xrange(25):
        passed = False
        while not passed:
            composite = genConfig(4, 0)
            description, clarification = composite.getDescription()
            if ''.join(description) not in descriptions and composite.normalize():
                descriptions.add(''.join(description))
                shapes.append(composite)
                passed = True
    with open(sys.argv[1], "w") as fout:
        for shape in shapes:
            shape.write_experiment(fout)
            fout.write("\n")

