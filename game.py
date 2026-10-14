import pygame

WIDTH, HEIGHT = 800, 600 #Dimensions of game.
DT = 1 / 60 #Time step to use
FIRERATE = 3


class Player:
    def __init__(self):
        self.position = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        self.angle = 0
        self.speed = 200
        self.turn_speed = 180
        self.radius = 15
        self.health = 5

    def update(self, movement, rotation, dt):
        self.angle = (self.angle + rotation * self.turn_speed * dt) % 360
        
        direction = pygame.Vector2(1, 0).rotate(self.angle)
        self.position += direction * movement * self.speed * dt

        self.position.x = max(
            self.radius, min(WIDTH - self.radius, self.position.x)
        )
        self.position.y = max(
            self.radius, min(HEIGHT - self.radius, self.position.y)
        )

class Bullet:
    def __init__(self, position, angle):
        self.position = position
        self.angle = angle
        self.speed = 500
        self.radius = 5
        self.hasHit = False
        self.oob = False

    def update(self, dt, players):
        direction = pygame.Vector2(1, 0).rotate(self.angle)
        self.position += direction * (self.speed * dt)
        
        if self.position.x > WIDTH or self.position.x < 0:
            self.oob = True
        if self.position.y > HEIGHT or self.position.y < 0:
            self.oob = True

        for player in players:
            dist = pow(pow(self.position.x - player.position.x, 2) +  pow(self.position.y - player.position.y, 2), .5)
            if dist < player.radius:
                player.health -= 1
                self.hasHit = True

def draw_player(screen, player):
    length = 20
    width = 20

    corners = [
        pygame.Vector2(0, -width/2),
        pygame.Vector2(length, -width/2),
        pygame.Vector2(length, width/2),
        pygame.Vector2(0, width/2)
    ]

    rotated_corners = [player.position + corner.rotate(player.angle) for corner in corners]
    pygame.draw.polygon(screen, "white", rotated_corners)

    pygame.draw.circle(screen, "skyblue", player.position, player.radius)

def draw_bullet(screen, bullet):
    pygame.draw.circle(screen, "red", bullet.position, bullet.radius)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tiny Blast!")
    clock = pygame.time.Clock()

    player = Player()
    timeDiff = 0.0
    running = True
    fireCD = 0
    bullets = []

    while running:
        elapsed = clock.tick(60) / 1000

        timeDiff += min(elapsed, 0.25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        keys = pygame.key.get_pressed()
        movement = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        rotation = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        fire = int(keys[pygame.K_SPACE])

        while timeDiff >= DT:
            if fireCD > 0: fireCD -= 1
            player.update(movement, rotation, DT)
            if fire == 1:
                if fireCD == 0:
                    newBullPosition = player.position + pygame.Vector2(1, 0).rotate(player.angle) * (player.radius + 1)
                    bullets.append(Bullet(newBullPosition, player.angle))
                    fireCD = 60/FIRERATE

            for bullet in bullets: 
                if bullet.hasHit or bullet.oob: 
                    bullets.remove(bullet)
                else:
                    bullet.update(DT, [player])
            timeDiff -= DT

        screen.fill((25, 25, 30))
        draw_player(screen, player)
        for bullet in bullets: draw_bullet(screen, bullet)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
