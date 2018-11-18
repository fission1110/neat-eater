import pygame
import sys
import time
import random
import neat
import os


class Sprite(object):
    def __init__(self,screen):
        self.color = pygame.Color('blue')
        self.energy = 1
        self.screen = screen
        self.surface = pygame.Surface((10,10))
        self.rect = self.surface.get_rect()
        self._draw()
        #self.set_loc((random.randrange(0,self.screen.get_width() - self.rect.width), random.randrange(0,self.screen.get_height() - self.rect.height)))
        self.set_loc((self.screen.get_width()//2, self.screen.get_height()//2))

    def move(self, direction):
        if len(direction) == 0:
            return False
        if 'u' in direction and 'd' in direction:
            return False
        if 'l' in direction and 'r' in direction:
            return False
        if 'u' in direction:
            if self.rect.top != 0:
                self.rect.move_ip(0,-1)
            else:
                return False
        elif 'd' in direction:
            if self.rect.bottom != self.screen.get_height():
                self.rect.move_ip(0,1)
            else:
                return False
        if 'l' in direction:
            if self.rect.left != 0:
                self.rect.move_ip(-1,0)
            else:
                return False
        elif 'r' in direction:
            if self.rect.right != self.screen.get_width():
                self.rect.move_ip(1,0)
            else:
                return False
        self._draw()
        return True

    def set_loc(self, loc):
        self.rect.left = loc[0]
        self.rect.top = loc[1]
        self._draw()

    def set_color(self, color):
        self.color = color
        self._draw()

    def _draw(self):
        self.surface.fill(self.color)

    def _blit(self):
        self.screen.blit(self.surface, self.rect)

class Bug(Sprite):
    def __init__(self, screen):
        super(Bug, self).__init__(screen)
        self.color = pygame.Color('red')
        self.energy = 0

    def check_eat(self, food_list):
        for i in range(0, len(food_list)-1):
            if self.rect.colliderect(food_list[i].rect):
                self.energy += food_list[i].energy
                food_list[i].set_color(pygame.Color('black'))
                food_list.pop(i)
                food_list = self.check_eat(food_list)
                break
        return food_list

class Food(Sprite):
    def __init__(self, screen):
        super(Food, self).__init__(screen)
        self.set_color(pygame.Color('white'))
        self.energy = 1
