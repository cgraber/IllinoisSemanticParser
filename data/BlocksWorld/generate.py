iters = 300
maxNumShapes = 3

import random
from random import randint, shuffle

COMMANDS = ["Create", "Construct", "Build", "Form"]
CONNECTORS = ["and", "with"]

ROW = 0
COL = 1
SQUARE = 2
RECT = 3



TOP = 0
LEFT = 1
RIGHT = 2
BOTTOM = 3
DIRECTIONS = ["top", "left", "right", "bottom"]

ORDINALS = ['', "first", "second", "third", "fourth", "fifth", "sixth"]

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
            self.logic += shape.logic
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
            self.logic += shape.logic
            if direction == LEFT:
                old = self.shapesOnSides[LEFT]
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

                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, shape.width - 1)
                    oldEnumVal = old.getEnum(offset, 0)
                elif shape.bottom > old.bottom:
                    #In this case, use the corner of the old shape
                    oldEnumVal = old.getEnum(0, 0)
                    newEnumVal = shape.getEnum(-1*offset, shape.width-1)
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, shape.width - 1)
                    oldEnumVal = old.getEnum(shape.bottom - old.top, 0)
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                self.logic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                self.logic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar), "enum(%s, %s, %d)"%(shape.var, newBlockVar, newEnumVal)]
                self.logic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                self.logic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar), "enum(%s, %s, %d)"%(old.var, oldBlockVar, oldEnumVal)]
                self.logic.append("spatial-rel(west, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))

            elif direction == RIGHT:
                old = self.shapesOnSides[RIGHT]
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
                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, 0)
                    oldEnumVal = old.getEnum(offset, old.width-1)
                elif shape.bottom > old.bottom:
                    #In this case, use the corner of the old shape
                    oldEnumVal = old.getEnum(0, old.width-1)
                    newEnumVal = shape.getEnum(-1*offset, 0)
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, 0)
                    oldEnumVal = old.getEnum(shape.bottom - old.top, old.width-1)
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                self.logic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                self.logic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar), "enum(%s, %s, %d)"%(shape.var, newBlockVar, newEnumVal)]
                self.logic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                self.logic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar), "enum(%s, %s, %d)"%(old.var, oldBlockVar, oldEnumVal)]
                self.logic.append("spatial-rel(east, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))
            elif direction == TOP:
                old = self.shapesOnSides[TOP]
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

                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(shape.height - 1, 0)
                    oldEnumVal = old.getEnum(0, offset)
                elif shape.right > old.right:
                    oldEnumVal = old.getEnum(0, 0)
                    newEnumVal = shape.getEnum(shape.height-1, -1*offset)
                else:
                    newEnumVal = shape.getEnum(shape.height - 1, shape.width - 1)
                    oldEnumVal = old.getEnum(0, shape.right - old.left)
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                self.logic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                self.logic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar), "enum(%s, %s, %d)"%(shape.var, newBlockVar, newEnumVal)]
                self.logic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                self.logic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar), "enum(%s, %s, %d)"%(old.var, oldBlockVar, oldEnumVal)]
                self.logic.append("spatial-rel(north, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))
            elif direction == BOTTOM:
                old = self.shapesOnSides[BOTTOM]
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

                # Additional Logic
                if offset >= 0:
                    newEnumVal = shape.getEnum(0, 0)
                    oldEnumVal = old.getEnum(old.height - 1, offset)
                elif shape.right > old.right:
                    oldEnumVal = old.getEnum(old.height-1, 0)
                    newEnumVal = shape.getEnum(0, -1*offset)
                else:
                    newEnumVal = shape.getEnum(0, shape.width - 1)
                    oldEnumVal = old.getEnum(old.height - 1, shape.right - old.left)
                newSpaceVar = getSpaceVar()
                oldSpaceVar = getSpaceVar()
                newBlockVar = getVar()
                oldBlockVar = getVar()
                self.logic += ["block(%s)"%newBlockVar, "location(%s)"%newSpaceVar]
                self.logic += ["block-location(%s, %s)"%(newBlockVar, newSpaceVar), "enum(%s, %s, %d)"%(shape.var, newBlockVar, newEnumVal)]
                self.logic += ["block(%s)"%oldBlockVar, "location(%s)"%oldSpaceVar]
                self.logic += ["block-location(%s, %s)"%(oldBlockVar, oldSpaceVar), "enum(%s, %s, %d)"%(old.var, oldBlockVar, oldEnumVal)]
                self.logic.append("spatial-rel(south, 0, %s, %s)"%(oldSpaceVar, newSpaceVar))
    def normalize(self):
        xShift = -1 * self.minX
        self.minX += xShift
        self.maxX += xShift
        yShift = -1 * self.minY
        self.minY += yShift
        self.maxY += yShift
        for shape in self.shapes:
            shape.left += xShift
            shape.right += xShift
            shape.top += yShift
            shape.bottom += yShift

    def getDescription(self):
        if self.description:
            return self.description

        self.description = random.choice(COMMANDS) + " "
        if len(self.shapes) == 1:
            self.description += self.shapes[0].description + "."
        elif len(self.shapes) == 2:
            self.description += self.shapes[0].description + " "
            self.description += random.choice(CONNECTORS) + " "
            self.description += self.shapes[1].description + " "
            self.description += "to its %s where "%DIRECTIONS[self.relations[0].direction]
            self.description += self.relations[0].description + "."
        else:
            self.description += "a structure consisting of "
            self.description += self.shapes[0].description + ", "
            for i in xrange(len(self.shapes)-2):
                #TODO: need intelligent resolution of its vs. more specific
                self.description += self.shapes[i+1].description + " to its %s where "%DIRECTIONS[self.relations[i].direction]
                self.description += self.relations[i].description + ", "
            self.description += "and "
            self.description += self.shapes[-1].description + " to its %s where "%DIRECTIONS[self.relations[-1].direction]
            self.description += self.relations[-1].description + "."
        return self.description


    def draw(self):
        self.normalize()
        result = [['O']*(self.maxX + 1) for i in xrange(self.maxY + 1)]
        for shape in self.shapes:
            shape.fillIn(result)
        return result

    def write(self, fout):
        result = self.draw()
        for row in xrange(len(result)):
            fout.write(" ".join(result[row]))
            fout.write("\n")
        fout.write("===\n")
        fout.write(' ^ '.join(self.logic))
        fout.write("\n")
        fout.write(self.getDescription())
        fout.write("\n")


class Row(Shape):
    DESCRIPTIONS = ["row", "horizontal line"]
    COMPOSITIONS = ["using %d blocks", "of size %d", "of length %d", "with %d blocks"]
    LEFT_CHOICES = ["the left end of %s", "%s's left end"]
    RIGHT_CHOICES = ["the right end of %s", "%s's right end"]
    def __init__(self, length):
        description = "a %s %s"%(random.choice(Row.DESCRIPTIONS), random.choice(Row.COMPOSITIONS)%length)
        #                   name, width, height, description
        Shape.__init__(self, "row", length, 1, description)
        self.logic = ['row(%s)'%self.var, 'width(%s, %d)'%(self.var, length)]

    def getLowerLeftDescription(self):
        return random.choice(Row.LEFT_CHOICES)
    def getUpperLeftDescription(self):
        return random.choice(Row.LEFT_CHOICES)
    def getLowerRightDescription(self):
        return random.choice(Row.RIGHT_CHOICES)
    def getUpperRightDescription(self):
        return random.choice(Row.RIGHT_CHOICES)


class Col(Shape):
    DESCRIPTIONS = ["column", "vertical line"]
    COMPOSITIONS = ["using %d blocks", "of size %d", "of height %d", "with %d blocks"]
    TOP_CHOICES = ["the top end of %s", "%s's top end"]
    BOTTOM_CHOICES = ["the bottom end of %s", "%s's bottom end"]
    def __init__(self, height):
        description = "a %s %s"%(random.choice(Col.DESCRIPTIONS), random.choice(Col.COMPOSITIONS)%height)
        #                   name, width, height, description
        Shape.__init__(self, "column", 1, height, description)
        self.logic = ['column(%s)'%self.var, 'height(%s, %d)'%(self.var, height)]

    def getLowerLeftDescription(self):
        return random.choice(Col.BOTTOM_CHOICES)
    def getUpperLeftDescription(self):
        return random.choice(Col.TOP_CHOICES)
    def getLowerRightDescription(self):
        return random.choice(Col.BOTTOM_CHOICES)
    def getUpperRightDescription(self):
        return random.choice(Col.TOP_CHOICES)

class Square(Shape):
    def __init__(self, dim):
        DESCRIPTIONS = ["a %d by %d square"%(dim, dim),
            "a square with sides of block length %d"%dim,
            "a %d block by %d block square"%(dim, dim)]
        description = random.choice(DESCRIPTIONS)
        #                   name, width, height, description
        Shape.__init__(self, "square", dim, dim, description)
        self.logic = ['square(%s)'%self.var, 'size(%s, %d)'%(self.var, dim)]

class Rect(Shape):
    def __init__(self, width, height):
        DESCRIPTIONS = ["a %d by %d rectangle"%(width, height), 
                        "a rectangle with a width of %d blocks and a height of %d blocks"%(width, height),
                        "a rectangle with a height of %d blocks and a width of %d blocks"%(height, width),
                        "a rectangle that is %d blocks wide and %d blocks tall"%(width, height),
                        "a rectangle that is %d blocks tall and %d blocks wide"%(height, width)]
        description = random.choice(DESCRIPTIONS)
        #                   name, width, height, description,
        Shape.__init__(self, "rectangle", width, height, description)
        self.logic = ['rectangle(%s)'%self.var, 'height(%s, %d)'%(self.var, height), 'width(%s, %d)'%(self.var, width)]

def randRow():
    length = randint(3,5)
    return Row(length)

def randCol():
    height = randint(3,5)
    return Col(height)

def randSquare():
    dim = randint(2,4)
    return Square(dim)

def randRect():
    length = randint(2,4)
    height = length
    while height == length:
        height = randint(2,4)
    return Rect(length, height)

genShape = [randRow, randCol, randSquare, randRect]

configs = []
descriptions = []
fout = open("data.txt", "w")

for i in xrange(iters):
    resetVars()
    numShapes = randint(1, maxNumShapes)
    composite = CompositeShape()
    for j in xrange(numShapes):
        newShape = genShape[randint(0,3)]()
        direction = randint(0,3)
        composite.addShape(newShape, direction)
    composite.write(fout)
    fout.write("\n")
fout.close()
'''
for i in xrange(2,6):
    current = Row(i)
    for description in current.descriptions:
        descriptions.append((current, description))
    current = Col(i)
    for description in current.descriptions:
        descriptions.append((current, description))

for i in xrange(2,5):
    current = Square(i)
    for description in current.descriptions:
        descriptions.append((current, description))

for i in xrange(2, 5):
    for j in xrange(2, 5):
        if i ==j:
            continue
        current = Rect(i,j)
        for description in current.descriptions:
            descriptions.append((current, description))
fout = open("data.txt", "w")
shuffle(descriptions)
for description in descriptions:
    description[0].write(fout)
    fout.write(description[1]+"\n\n")
'''
