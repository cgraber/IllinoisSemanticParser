TRAIN_SIZE = 1400
TEST_SIZE = 600
maxNumShapes = 3

import random, sys, os
import gen_images
from random import randint, shuffle

COMMANDS = ["Create", "Construct", "Build", "Form"]
CONNECTORS = ["and", "with"]
NEXT = ["Next", "Then", "After that"]

ROW = 0
COL = 1
SQUARE = 2
RECT = 3



TOP = 0
LEFT = 1
RIGHT = 2
BOTTOM = 3
DIRECTIONS = ["top", "left", "right", "bottom"]

ORDINALS = ['', "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

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
    def __init__(self, direction, second, first, offset):
        self.direction = direction
        self.second = second
        self.first = first
        self.offset = offset
        self.description = "" 

        # Build the description
        # Recipe: [SECOND REFERENCE] [RELATIVE LOCATION] [FIRST REFERENCE]
        # note: for now, it is assumed that the second shape was just mentioned.
        if first.ind != 0:
            firstName = "the %s %s"%(ORDINALS[first.ind], first.name)
        else:
            firstName = "the %s"%first.name
        if second.ind != 0:
            secondName = "%s %s"%(random.choice(["this", "the new"]), second.name)
        else:
            secondName = "the %s"%second.name
        if self.direction == LEFT:
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
            else:
                self.description += random.choice(Relation.NEXT_CHOICES) + " "

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

        elif self.direction == RIGHT:
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
            else:
                self.description += random.choice(Relation.NEXT_CHOICES) + " "

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
                    self.description += self.first.getUpperRightDescription()%firstName
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
            else:
                self.description += random.choice(Relation.NEXT_CHOICES) + " "

            # FIRST REFERENCE
            if left:
                if second.left == first.left:
                    self.description += first.getUpperLeftDescription()%firstName
                elif second.left == first.right:
                    self.description += first.getUpperRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(TOP, offset, firstName)
            elif middle:
                self.description += self.first.getUpperLeftDescription()%firstName
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
            else:
                self.description += random.choice(Relation.NEXT_CHOICES) + " "

            # FIRST REFERENCE
            if left:
                if second.left == first.left:
                    self.description += first.getLowerLeftDescription()%firstName
                elif second.left == first.right:
                    self.description += first.getLowerRightDescription()%firstName
                else:
                    self.description += first.getEdgeDescription(TOP, offset, firstName)
            elif middle:
                self.description += self.first.getLowerLeftDescription()%firstName
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

    def addShape(self, shape, direction):
        if not self.shapes:
            self.shapes = [shape]
            #The following keeps track of which shape is on each side.
            self.shapesOnSides = [shape]*4
            self.maxX = shape.right
            self.minX = 0
            self.maxY = shape.bottom
            self.minY = 0
            self.logic.append(shape.getLogic())
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
            newLogic += shape.getLogic()
            #newLogic += shape.getCondensedLogic() #TODO: THIS IS WHAT YOU CHANGED
            if direction == LEFT:
                old = self.shapesOnSides[LEFT]
                #newLogic += old.getCondensedLogic()
                offset = randint(-1*(shape.height - 1), old.height - 1)
                shape.right = old.left - 1
                shape.left = old.left - shape.width
                shape.top = old.top + offset
                shape.bottom = old.top + offset + shape.height - 1
                self.relations.append(Relation(LEFT, shape, old, offset))
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
                newLogic.append("left(%s, %s)"%(shape.var, old.var))
                newLogic = []
                #newLogic += shape.getCondensedLogic()
                #newLogic += old.getCondensedLogic()
                self.logic.append(newLogic)

                # Additional Logic
                if offset >= 0:
                    #newEnumVal = shape.getEnum(0, shape.width - 1)
                    newRow = 0
                    newCol = shape.width-1
                    #newEnumLogic = "upper_right(%s, %s)"
                    #oldEnumVal = old.getEnum(offset, 0)
                    oldRow = offset
                    oldCol = 0
                    if offset == 0:
                        oldEnumLogic = "upper_left(%s, %s)"
                    elif shape.top == old.bottom:
                        oldEnumLogic = "lower_left(%s, %s)"
                    else:
                        oldEnumLogic = "left_side(%s, %s, "+str(offset)+")"
                elif shape.bottom > old.bottom:
                    #In this case, use the corner of the old shape
                    oldEnumVal = old.getEnum(0, 0)
                    oldRow = 0
                    oldCol = 0
                    oldEnumLogic = "upper_left(%s, %s)"
                    newEnumVal = shape.getEnum(-1*offset, shape.width-1)
                    newRow = -1*offset
                    newCol = shape.width-1
                    newEnumLogic = "right_side(%s, %s, "+str(-1*offset)+")"
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, shape.width - 1)
                    newRow = shape.height-1
                    newCol = shape.width-1
                    newEnumLogic = "lower_right(%s, %s)"
                    oldEnumVal = old.getEnum(shape.bottom - old.top, 0)
                    oldRow = shape.bottom - old.top
                    oldCol = 0
                    if shape.bottom == old.bottom:
                        oldEnumLogic = "lower_left(%s, %s)"
                    elif shape.bottom == old.top:
                        oldEnumLogic = "upper_left(%s, %s)"
                    else:
                        oldEnumLogic = "left_side(%s, %s, "+str(shape.bottom-old.top) + ")"
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(shape.var, newBlockVar, newRow), "col-ind(%s, %s, %d)"%(shape.var, newBlockVar, newCol)]
                newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldRow), "col-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldCol)]
                newLogic.append("spatial-rel(west, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))

            elif direction == RIGHT:
                old = self.shapesOnSides[RIGHT]
                #newLogic += old.getCondensedLogic()
                offset = randint(-1*(shape.height - 1), old.height - 1)
                shape.left = old.right + 1
                shape.right = old.right + shape.width
                shape.top = old.top + offset
                shape.bottom = old.top + offset + shape.height - 1
                self.relations.append(Relation(RIGHT, shape, old, offset))
                if shape.right >= self.maxX:
                    self.maxX = shape.right
                    self.shapesOnSides[RIGHT] = shape
                if shape.top <= self.minY:
                    self.minY = shape.top
                    self.shapesOnSides[TOP] = shape
                if shape.bottom >= self.maxY:
                    self.maxY = shape.bottom
                    self.shapesOnSides[BOTTOM] = shape
                newLogic.append("right(%s, %s)"%(shape.var, old.var))
                newLogic = []
                #newLogic += shape.getCondensedLogic()
                #newLogic += old.getCondensedLogic()
                self.logic.append(newLogic)

                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, 0)
                    newRow = 0
                    newCol = 0
                    newEnumLogic = "upper_left(%s, %s)"
                    oldEnumVal = old.getEnum(offset, old.width-1)
                    oldRow = offset
                    oldCol = old.width-1
                    if offset == 0:
                        oldEnumLogic = "upper_right(%s, %s)"
                    elif shape.top == old.bottom:
                        oldEnumLogic = "lower_right(%s, %s)"
                    else:
                        oldEnumLogic = "right_side(%s, %s, "+str(offset) +")"
                elif shape.bottom > old.bottom:
                    #In this case, use the corner of the old shape
                    oldEnumVal = old.getEnum(0, old.width-1)
                    oldRow = 0
                    oldCol = old.width-1
                    oldEnumLogic = "upper_right(%s, %s)"
                    newEnumVal = shape.getEnum(-1*offset, 0)
                    newRow = -1*offset
                    newCol = 0
                    newEnumLogic = "left_side(%s, %s, "+str(-1*offset)+")"
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, 0)
                    newRow = shape.height-1
                    newCol = 0
                    newEnumLogic = "lower_left(%s, %s)"
                    oldEnumVal = old.getEnum(shape.bottom - old.top, old.width-1)
                    oldRow = shape.bottom - old.top
                    oldCol = old.width-1
                    if shape.bottom == old.bottom:
                        oldEnumLogic = "lower_right(%s, %s)"
                    elif shape.bottom == old.top:
                        oldEnumLogic = "upper_right(%s, %s)"
                    else:
                        oldEnumLogic = "right_side(%s, %s, "+str(shape.bottom-old.top)+")"
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(shape.var, newBlockVar, newRow), "col-ind(%s, %s, %d)"%(shape.var, newBlockVar, newCol)]
                newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldRow), "col-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldCol)]
                newLogic.append("spatial-rel(east, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))
            elif direction == TOP:
                old = self.shapesOnSides[TOP]
                #newLogic += old.getCondensedLogic()
                offset = randint(-1*(shape.width-1), old.width - 1)
                shape.bottom = old.top - 1
                shape.top = old.top - shape.height
                shape.left = old.left + offset
                shape.right = old.left + offset + shape.width - 1
                self.relations.append(Relation(TOP, shape, old, offset))
                if shape.left <= self.minX:
                    self.minX = shape.left
                    self.shapesOnSides[LEFT] = shape
                if shape.right >= self.maxX:
                    self.maxX = shape.right
                    self.shapesOnSides[RIGHT] = shape
                if shape.top <= self.minY:
                    self.minY = shape.top
                    self.shapesOnSides[TOP] = shape
                newLogic.append("top(%s, %s)"%(shape.var, old.var))
                newLogic = []
                #newLogic += shape.getCondensedLogic()
                #newLogic += old.getCondensedLogic()
                self.logic.append(newLogic)

                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(shape.height - 1, 0)
                    newRow = shape.height-1
                    newCol = 0
                    newEnumLogic = "lower_left(%s, %s)"
                    oldEnumVal = old.getEnum(0, offset)
                    oldRow = 0
                    oldCol = offset
                    if offset == 0:
                        oldEnumLogic = "upper_left(%s, %s)"
                    elif shape.left == old.right:
                        oldEnumLogic = "upper_right(%s, %s)"
                    else:
                        oldEnumLogic = "top_side(%s, %s, "+str(offset)+")"
                elif shape.right > old.right:
                    oldEnumVal = old.getEnum(0, 0)
                    oldRow = 0
                    oldCol = 0
                    oldEnumLogic = "upper_left(%s, %s)"
                    newEnumVal = shape.getEnum(shape.height-1, -1*offset)
                    newRow = shape.height-1
                    newCol = -1*offset
                    newEnumLogic = "bottom_side(%s, %s, "+str(-1*offset)+")"
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, shape.width - 1)
                    newRow = shape.height-1
                    newCol = shape.width-1
                    newEnumLogic = "lower_right(%s, %s)"
                    oldEnumVal = old.getEnum(0, shape.right - old.left)
                    oldRow = 0
                    oldCol = shape.right-old.left
                    if old.right == shape.right:
                        oldEnumLogic = "upper_right(%s, %s)"
                    elif shape.right == old.left:
                        oldEnumLogic = "upper_left(%s, %s)"
                    else:
                        oldEnumLogic = "top_side(%s, %s, "+str(shape.right - old.left)+")"
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(shape.var, newBlockVar, newRow), "col-ind(%s, %s, %d)"%(shape.var, newBlockVar, newCol)]
                newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldRow), "col-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldCol)]
                newLogic.append("spatial-rel(north, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))
            elif direction == BOTTOM:
                old = self.shapesOnSides[BOTTOM]
                #newLogic += old.getCondensedLogic()
                offset = randint(-1*(shape.width-1), old.width - 1)
                shape.top = old.bottom + 1
                shape.bottom = old.bottom + shape.height
                shape.left = old.left + offset
                shape.right = old.left + offset + shape.width - 1
                self.relations.append(Relation(BOTTOM, shape, old, offset))
                if shape.left <= self.minX:
                    self.minX = shape.left
                    self.shapesOnSides[LEFT] = shape
                if shape.right >= self.maxX:
                    self.maxX = shape.right
                    self.shapesOnSides[RIGHT] = shape
                if shape.bottom >= self.maxY:
                    self.maxY = shape.bottom
                    self.shapesOnSides[BOTTOM] = shape

                newLogic.append("bottom(%s, %s)"%(shape.var, old.var))
                newLogic = []
                #newLogic += shape.getCondensedLogic()
                #newLogic += old.getCondensedLogic()
                self.logic.append(newLogic)

                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, 0)
                    newRow = 0
                    newCol = 0
                    newEnumLogic = "upper_left(%s, %s)"
                    oldEnumVal = old.getEnum(old.height - 1, offset)
                    oldRow = old.height-1
                    oldCol = offset
                    if offset == 0:
                        oldEnumLogic = "lower_left(%s, %s)"
                    elif shape.left == old.right:
                        oldEnumLogic = "lower_right(%s, %s)"
                    else:
                        oldEnumLogic = "bottom_side(%s, %s, "+str(offset)+")"
                elif shape.right > old.right:
                    oldEnumVal = old.getEnum(old.height-1, 0)
                    oldRow = old.height-1
                    oldCol = 0
                    oldEnumLogic = "lower_left(%s, %s)"
                    newEnumVal = shape.getEnum(0, -1*offset)
                    newRow = 0
                    newCol = -1*offset
                    newEnumLogic = "top_side(%s, %s, "+str(-1*offset)+")"
                else:
                    newEnumVal = shape.getEnum(0, shape.width - 1)
                    newRow = 0
                    newCol = shape.width-1
                    newEnumLogic = "upper_right(%s, %s)"
                    oldEnumVal = old.getEnum(old.height - 1, shape.right - old.left)
                    oldRow = old.height-1
                    oldCol = shape.right-old.left
                    if old.right == shape.right:
                        oldEnumLogic = "lower_right(%s, %s)"
                    elif shape.right == old.left:
                        oldEnumLogic = "lower_left(%s, %s)"
                    else:
                        oldEnumLogic = "bottom_side(%s, %s, "+str(shape.right - old.left)+")"
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                newLogic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                newLogic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(shape.var, newBlockVar, newRow), "col-ind(%s, %s, %d)"%(shape.var, newBlockVar, newCol)]
                newLogic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                newLogic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar), oldEnumLogic%(old.var, oldBlockVar)]
                newLogic += ["row-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldRow), "col-ind(%s, %s, %d)"%(old.var, oldBlockVar, oldCol)]
                newLogic.append("spatial-rel(south, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))

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
        if num == 1:
            self.description.append("Do not put a block in the space at row %d and column %d."%(restricted[0][1], restricted[0][0]))
        elif num == 2:
            self.description.append("Do not put blocks in the spaces at row %d and column %d and at row %d and column %d."%(restricted[0][1], restricted[0][0], restricted[1][1], restricted[1][0]))
        else:
            result = "Do not put blocks in "
            for i in xrange(num-1):
                result += "the space at row %d and column %d, "%(restricted[i][1], restricted[i][0])
            result += "and the space at row %d and column %d."%(restricted[-1][1], restricted[-1][0])
            self.description.append(result)


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

    def addImmobileBlocks(self):
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
            new_formula.append("immobile_block(%d, %d)"%(xConst, yConst))
        self.logic.append(new_formula)
        self.getDescription()
        if num == 1:
            self.description.append("There is an immobile block at row %d and column %d."%(restricted[0][1], restricted[0][0]))
        elif num == 2:
            self.description.append("There are immobile blocks at row %d and column %d and at row %d and column %d."%(restricted[0][1], restricted[0][0], restricted[1][1], restricted[1][0]))
        else:
            result = "There are immobile blocks "
            for i in xrange(num-1):
                result += "at row %d and column %d, "%(restricted[i][1], restricted[i][0])
            result += "and at row %d and column %d."%(restricted[-1][1], restricted[-1][0])
            self.description.append(result)

    def normalize(self):
        xShift = -1 * self.minX
        self.minX += xShift
        self.maxX += xShift
        yShift = -1 * self.minY
        self.minY += yShift
        self.maxY += yShift

        if self.maxX >= gen_images.GRID_WIDTH or self.maxY >= gen_images.GRID_HEIGHT:
            return False
        if self.maxX < gen_images.GRID_WIDTH - 1:
            xShift += (gen_images.GRID_WIDTH - self.maxX - 1) / 2
        if self.maxY < gen_images.GRID_HEIGHT - 1:
            yShift += (gen_images.GRID_HEIGHT - self.maxY - 1) / 2

        for shape in self.shapes:
            shape.left += xShift
            shape.right += xShift
            shape.top += yShift
            shape.bottom += yShift
        return True

    def getDescription(self):
        if self.description:
            return self.description

        self.description = []
        if len(self.shapes) == 1:
            description = random.choice(COMMANDS) + " "
            description += self.shapes[0].description + "."
            self.description.append(description)
        elif len(self.shapes) == 2:
            description = random.choice(COMMANDS) + " "
            description += self.shapes[0].description + "."
            self.description.append(description)
            description = "%s, "%random.choice(NEXT)
            description += "%s "%random.choice(COMMANDS).lower()
            if self.shapes[0].__class__ == self.shapes[1].__class__:
                description += "another one "
                description += "to the %s of the first one." %(DIRECTIONS[self.relations[0].direction])
                #description += self.shapes[1].size_description + " to the %s of the first one." %(DIRECTIONS[self.relations[0].direction])
            else:
                name = "the %s"%self.shapes[0].name
                #description += "a %s to the %s of %s."%(self.shapes[1].name, DIRECTIONS[self.relations[0].direction], name)
                description += self.shapes[1].description + " to the %s of %s." %(DIRECTIONS[self.relations[0].direction], name)
            self.description.append(description)
            self.description.append("Ensure that %s."%self.relations[0].description)

        return self.description


    def draw(self):
        result = [['O']*(gen_images.GRID_WIDTH) for i in xrange(gen_images.GRID_HEIGHT)]
        for shape in self.shapes:
            shape.fillIn(result)
        return result

    def write(self, fout):
        result = self.draw()
        for row in xrange(len(result)):
            fout.write(" ".join(result[row]))
            fout.write("\n")
        fout.write("===\n")
        description = self.getDescription()
        for i in xrange(len(self.description)):
            fout.write(' ^ '.join(self.logic[i]))
            fout.write("\n")
            fout.write(description[i])
            fout.write("\n")

    def draw_to_file(self, file_path):
        shape = self.draw()
        gen_images.draw_shape_to_file(shape, file_path)


class Row(Shape):
    DESCRIPTIONS = ["row", "horizontal line"]
    COMPOSITIONS = ["using %d blocks", "of size %d", "of length %d", "with %d blocks"]
    LEFT_CHOICES = ["the left end of %s", "%s's left end"]
    RIGHT_CHOICES = ["the right end of %s", "%s's right end"]
    def __init__(self, length):
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


class Col(Shape):
    DESCRIPTIONS = ["column", "vertical line"]
    COMPOSITIONS = ["using %d blocks", "of size %d", "of height %d", "with %d blocks"]
    TOP_CHOICES = ["the top end of %s", "%s's top end"]
    BOTTOM_CHOICES = ["the bottom end of %s", "%s's bottom end"]
    def __init__(self, height):
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

class Square(Shape):
    def __init__(self, dim):
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

class Rect(Shape):
    def __init__(self, width, height):
        DESCRIPTIONS = ["a %d by %d rectangle"%(width, height), 
                        "a rectangle with a width of %d blocks and a height of %d blocks"%(width, height),
                        "a rectangle with a height of %d blocks and a width of %d blocks"%(height, width),
                        "a rectangle that is %d blocks wide and %d blocks tall"%(width, height),
                        "a rectangle that is %d blocks tall and %d blocks wide"%(height, width)]
        self.size_description = random.choice(DESCRIPTIONS[1:])[12:]
        description = random.choice(DESCRIPTIONS)
        #                   name, width, height, description,
        Shape.__init__(self, "rectangle", width, height, description)

    def getLogic(self):
        return ['rectangle(%s)'%self.var, 'height(%s, %d)'%(self.var, self.height), 'width(%s, %d)'%(self.var, self.width)]
    def getCondensedLogic(self):
        return ['rectangle(%s)'%self.var]

def randRow():
    length = randint(3,10)
    return Row(length)

def randCol():
    height = randint(3,10)
    return Col(height)

def randSquare():
    dim = randint(2,10)
    return Square(dim)

def randRect():
    length = randint(2,10)
    height = length
    while height == length:
        height = randint(2,10)
    return Rect(length, height)

genShape = {ROW:randRow, COL:randCol, SQUARE:randSquare, RECT:randRect}

configs = []
descriptions = []
if len(sys.argv) != 4:
    print "Usage: python generate.py <train output name> <test output name> <image dir>"
    sys.exit(1)

shapes = []
descriptions = set()
while len(shapes) < TRAIN_SIZE + TEST_SIZE:
    resetVars()
    numShapes = randint(1, 2)
    composite = CompositeShape()
    prevShapeNum = None
    for j in xrange(numShapes):
        newShapeNum = randint(0,3)
        if newShapeNum == prevShapeNum:
            # We want the other shape to have the same size
            if newShapeNum == ROW:
                newShape = Row(prevShape.width)
            elif newShapeNum == COL:
                newShape = Col(prevShape.height)
            elif newShapeNum == SQUARE:
                newShape = Square(prevShape.width)
            else:
                newShape = Rect(prevShape.width, prevShape.height)
        else:
            newShape = genShape[newShapeNum]()
        direction = randint(0,3)
        composite.addShape(newShape, direction)
        prevShapeNum = newShapeNum
        prevShape = newShape
    constr = randint(0,3)
    if constr == 1:
        composite.addRandomConstraint()
    elif constr == 2:
        composite.addStartBlocks()
    elif constr == 3:
        composite.addImmobileBlocks()
    if ''.join(composite.getDescription()) not in descriptions and composite.normalize():
        descriptions.add(''.join(composite.getDescription()))
        shapes.append(composite)

train_dir = os.path.join(sys.argv[3], "train")
test_dir = os.path.join(sys.argv[3], "test")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

with open(sys.argv[1], "w") as fout:
    for i in xrange(TRAIN_SIZE):
        shape = shapes[i]
        shape.write(fout)
        fout.write("\n")
        shape.draw_to_file(os.path.join(train_dir, "train_%d.png"%i))
with open(sys.argv[2], "w") as fout:
    for i in xrange(TRAIN_SIZE, len(shapes)):
        shape = shapes[i]
        shape.write(fout)
        fout.write("\n")
        shape.draw_to_file(os.path.join(test_dir, "test_%d.png"%(i-TRAIN_SIZE)))
