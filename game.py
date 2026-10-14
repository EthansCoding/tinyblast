import pygame

WIDTH, HEIGHT = 800, 600 #Dimensions of game.
DT = 1 / 60 #Time step to use
FIRERATE = 3


class Player:
    def __init__(self, position=None):
        self.position = pygame.Vector2(
            position if position is not None else (WIDTH / 2, HEIGHT / 2)
        )
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


class GameHandler:
    def __init__(self, display_screen=True):
        self.players = []
        self.bullets = []
        self.screen = None
        self._inputs = {}
        self._fire_cooldowns = {}
        self._time_diff = 0.0

        if display_screen:
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Tiny Blast!")

    def create_player(self, position=None):
        player = Player(position)
        self.players.append(player)
        self._inputs[player] = (0, 0, False)
        self._fire_cooldowns[player] = 0
        return player

    def set_input(self, player, movement=0, rotation=0, fire=False):
        if player not in self._inputs:
            raise ValueError("Player must be created by this game handler")
        self._inputs[player] = (movement, rotation, fire)

    def _step(self):
        for player in self.players:
            movement, rotation, fire = self._inputs[player]
            self._fire_cooldowns[player] = max(
                0, self._fire_cooldowns[player] - 1
            )

            player.update(movement, rotation, DT)
            if fire and self._fire_cooldowns[player] == 0:
                direction = pygame.Vector2(1, 0).rotate(player.angle)
                position = player.position + direction * (player.radius + 1)
                self.bullets.append(Bullet(position, player.angle))
                self._fire_cooldowns[player] = 1 / (DT * FIRERATE)

        for bullet in self.bullets:
            bullet.update(DT, self.players)
        self.bullets = [
            bullet for bullet in self.bullets
            if not bullet.hasHit and not bullet.oob
        ]

    def update(self, elapsed=DT):
        if elapsed < 0:
            raise ValueError("Elapsed time must not be negative")
        self._time_diff += min(elapsed, 0.25)
        while self._time_diff >= DT:
            self._step()
            self._time_diff -= DT

        if self.screen is not None:
            self._draw()

        return (
            [(player.position.x, player.position.y) for player in self.players],
            [(bullet.position.x, bullet.position.y) for bullet in self.bullets],
        )

    def _draw(self):
        self.screen.fill((25, 25, 30))
        for player in self.players:
            draw_player(self.screen, player)
        for bullet in self.bullets:
            draw_bullet(self.screen, bullet)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None


def main():
    game = GameHandler(display_screen=True)
    player = game.create_player()
    clock = pygame.time.Clock()

    try:
        while True:
            elapsed = clock.tick(60) / 1000
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                break

            keys = pygame.key.get_pressed()
            game.set_input(
                player,
                movement=int(keys[pygame.K_w]) - int(keys[pygame.K_s]),
                rotation=int(keys[pygame.K_d]) - int(keys[pygame.K_a]),
                fire=bool(keys[pygame.K_SPACE]),
            )
            game.update(elapsed)
    finally:
        game.close()


if __name__ == "__main__":
    main()
