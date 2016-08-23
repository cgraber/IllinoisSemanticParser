import pygame
import pygame.locals

BLOCK_SIZE = 30
GRID_WIDTH = 15
GRID_HEIGHT = 15
GRID_TILE_PATH = "./ref_images/grid_tile.png"
BLOCK_TILE_PATH = "./ref_images/block.png"


pygame.init()
pygame.display.set_mode((GRID_WIDTH*BLOCK_SIZE, GRID_HEIGHT*BLOCK_SIZE))
grid_tile = pygame.image.load(GRID_TILE_PATH).convert()
block_tile = pygame.image.load(BLOCK_TILE_PATH).convert()

def draw_grid(screen, grid_tile):
    for x in xrange(GRID_WIDTH):
        for y in xrange(GRID_HEIGHT):
            screen.blit(grid_tile, (x*BLOCK_SIZE, y*BLOCK_SIZE))

def draw_block(screen, block_tile, row, col):
    screen.blit(block_tile, (col*BLOCK_SIZE, row*BLOCK_SIZE))

def draw_shape(shape, screen, block_tile):
    for row in xrange(GRID_HEIGHT):
        for col in xrange(GRID_WIDTH):
            if shape[row][col] == 'X':
                draw_block(screen, block_tile, row, col)

def draw_shape_to_file(shape, file_path):
    screen = pygame.display.set_mode((GRID_WIDTH*BLOCK_SIZE, GRID_HEIGHT*BLOCK_SIZE))
    draw_grid(screen, grid_tile)
    draw_shape(shape, screen, block_tile)
    pygame.image.save(screen, file_path)
    

if __name__=='__main__':
    screen = pygame.display.set_mode((GRID_WIDTH*BLOCK_SIZE, GRID_HEIGHT*BLOCK_SIZE))
    test_shape = [
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','O','O','O','X','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','X','O','O','O','X','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','X','O','O','X','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','X','X','X','X','X','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','X','X','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','X','X','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                  ['O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],
                 ]
    draw_shape(test_shape, screen, block_tile)
    pygame.display.flip()
    while pygame.event.wait().type != pygame.locals.QUIT:
        pass
