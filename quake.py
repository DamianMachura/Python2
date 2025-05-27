import pygame
import math
import random

# Inicjalizacja Pygame
pygame.init()

# Ustawienia ekranu
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


# generuje mape
class Map_Gen:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def generate_maze(self):
        maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        start_x, start_y = 1, 1
        maze[start_y][start_x] = 0
        
        # Kierunki (góra, dół, lewo, prawo)
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        
        def carve_path(x, y):
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and maze[ny][nx] == 1:
                    maze[y + dy // 2][x + dx // 2] = 0  # Usuwamy ścianę między komórkami
                    maze[ny][nx] = 0  # Tworzymy przejście
                    carve_path(nx, ny)
        
        carve_path(start_x, start_y)
        
        return maze
mp = Map_Gen(100,100)
MAP = mp.generate_maze()
MAP_WIDTH, MAP_HEIGHT = len(MAP[0]), len(MAP)
TILE_SIZE = 100
SCALE = WIDTH // 120  # Liczba promieni

class Bullet:
    def __init__(self, x, y, angle):
        self.x = x 
        self.y = y
        self.angle = math.atan2(300 - y, 400 - x)
        self.speed = 5  
        self.active = True

    def move(self):
        if self.active:
            self.x += math.cos(self.angle) * self.speed
            self.y += math.sin(self.angle) * self.speed

    def check_collision(self, map_data):
        col, row = int(self.x / TILE_SIZE), int(self.y / TILE_SIZE)
        if 0 <= row < len(map_data) and 0 <= col < len(map_data[0]):
            if map_data[row][col] == 1:  # Trafienie w ścianę
                self.active = False  # Zatrzymaj pocisk
                return True
        return False

    def draw(self):
        if self.active:
            pygame.draw.circle(screen, (100, 100, 100, 100), (self.x, self.y), 6)
            pygame.draw.circle(screen, (0, 255, 0), (self.x, self.y), 5)

class Player:
    def __init__(self, x, y, angle, speed):
        self._x = x
        self._y = y
        self._angle = angle
        self._speed = speed
        self._fov = math.pi / 3  # Pole widzenia
        self._num_rays = 120  
        self._max_depth = 8 
        self._collision_offset = 0.2
        self.ammo = 100  # Liczba pocisków, którą gracz ma
        self.bullets = []  # Lista pocisków, które zostały wystrzelone

    def move(self, keys, map_data):
        new_x, new_y = self._x, self._y
        if keys[pygame.K_w]:
            new_x += math.cos(self._angle) * self._speed
            new_y += math.sin(self._angle) * self._speed
        if keys[pygame.K_s]:
            new_x -= math.cos(self._angle) * self._speed
            new_y -= math.sin(self._angle) * self._speed

        if not self.check_collision(new_x + self._collision_offset, new_y + self._collision_offset, map_data):
            self._x, self._y = new_x, new_y

        if keys[pygame.K_a]:
            self._angle -= 0.05
        if keys[pygame.K_d]:
            self._angle += 0.05

    def check_collision(self, x, y, map_data):
        col, row = int(x), int(y)
        if 0 <= row < len(map_data) and 0 <= col < len(map_data[0]):
            return map_data[row][col] == 1
        return True

    def shoot(self):
        if self.ammo > 0:
            self.bullets.append(Bullet(self._x * TILE_SIZE, self._y * TILE_SIZE, self._angle))
            self.ammo -= 1

    def get_position(self):
        return self._x, self._y

    def get_angle(self):
        return self._angle

class Raycaster:
    def __init__(self, player, map_data):
        self.player = player
        self.map_data = map_data

    def cast_rays(self):
        start_angle = self.player.get_angle() - self.player._fov / 2
        for ray in range(self.player._num_rays):
            angle = start_angle + (ray / self.player._num_rays) * self.player._fov
            sin_a, cos_a = math.sin(angle), math.cos(angle)

            for depth in range(1, self.player._max_depth * TILE_SIZE, 5):
                target_x = self.player.get_position()[0] * TILE_SIZE + depth * cos_a
                target_y = self.player.get_position()[1] * TILE_SIZE + depth * sin_a
                col, row = int(target_x / TILE_SIZE), int(target_y / TILE_SIZE)
                if 0 <= row < len(self.map_data) and 0 <= col < len(self.map_data[0]) and self.map_data[row][col] == 1:
                    depth *= math.cos(self.player.get_angle() - angle)  # Korekta dystansu
                    wall_height = min(HEIGHT, TILE_SIZE * HEIGHT / (depth + 0.1))
                    color = (255 - min(depth, 255), 255 - min(depth, 255), 255 - min(depth, 255))
                    pygame.draw.rect(screen, color, (ray * SCALE, HEIGHT//2 - wall_height // 2, SCALE, wall_height))
                    break

def game_loop():
    player = Player(3, 3, 0, 0.05)
    raycaster = Raycaster(player, MAP)
    running = True

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    player.shoot()

        keys = pygame.key.get_pressed()
        player.move(keys, MAP)

        for bullet in player.bullets[:]:
            bullet.move()
            if bullet.check_collision(MAP):
                player.bullets.remove(bullet)
            bullet.draw()

        raycaster.cast_rays()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

game_loop()
