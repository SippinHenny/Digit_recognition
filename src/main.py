import pygame
import sys
from ia import *

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Digit recognition app")

# varible to store the number recognized
num = None


# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0,255,0)
grey = (128,128,128)

# check button
button_rect = pygame.Rect(20, 490, 100, 100)  # x, y, width, height
button_color = green

# reset button
button_rect2 = pygame.Rect(690, 490, 100, 100)  # x, y, width, height
button_color2 = red

#rect design
rect = pygame.Rect(0, 480, 800, 120)  # x, y, width, height
rect_color = grey

#cicrle location
circle = pygame.Rect(15, 15, 30, 30)  # x, y, width, height

# Set up drawing variables
drawing = False
radius = 13
color = white

#fill the screen in black
screen.fill(black)

# Linear interpolation function
def lerp(start, end, alpha):
    return int((1 - alpha) * start + alpha * end)

font = pygame.font.Font(None, 36)  

def draw_circle(start, end):
    for alpha in range(0, 101, 5):
        x = lerp(start[0], end[0], alpha / 100)
        y = lerp(start[1], end[1], alpha / 100)
        pygame.draw.circle(screen, color, (x, y), radius)


# check if the mouse is over the button
def is_mouse_over_button(pos, button_rect):
    return button_rect.collidepoint(pos)

# check if the mouse is over one of the buttons
def is_on_buttons(pos):
    if button_rect.collidepoint(pos) or button_rect2.collidepoint(pos) or circle.collidepoint(pos):
        return True
    else:
        return False

# Main game loop
last_pos = None
# Load the pre-trained model if it exists, otherwise train the model
if os.path.isfile('mnist_complex_cnn_model.pth'):
    model_loaded = True
else:
    model_loaded = False
    launch()

# Main game loop
while True:
    for event in pygame.event.get():
        # Quit game
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # condition to check if one of the button a been pushed
        elif event.type == pygame.MOUSEBUTTONDOWN and is_on_buttons(event.pos) :
            if event.button == 1:
                if is_mouse_over_button(event.pos, button_rect):
                    pygame.image.save(screen, "drawing.png")
                    num = recognize_digits()
                    screen.fill(black)
                    last_pos = None
                if is_mouse_over_button(event.pos, button_rect2):
                    screen.fill(black)
                    num = None
                    last_pos = None
                if is_mouse_over_button(event.pos, circle):
                    pygame.quit()
                    sys.exit()
        # xondition to check if the left mouse button is UP
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            drawing = False
            last_pos = None
        # condition to check if the left mouse button is DOWN
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            drawing = True
        # condition to check if the mouse is moving
        elif event.type == pygame.MOUSEMOTION and drawing:
            if last_pos:
                tick = 1
                draw_circle(last_pos, event.pos)
            else:
                tick = 1
                pygame.draw.circle(screen, color, event.pos, radius)
            last_pos = event.pos

    # Update
    pygame.display.flip()
    
    # Draw rect
    pygame.draw.rect(screen, rect_color, rect)

    # Draw button check
    pygame.draw.rect(screen, button_color, button_rect)

    # Draw button reset
    pygame.draw.rect(screen, button_color2, button_rect2)

    #draw Quit button
    pygame.draw.circle(screen, red, (30, 30), 20)

    if num != None:
        text = font.render(str(num), True, white)
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)
        

    # Draw button check text
    button_text = font.render("check", True, white)
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)

    # Draw button reset text
    button_text2 = font.render("reset", True, white)
    text_rect2 = button_text2.get_rect(center=button_rect2.center)
    screen.blit(button_text2, text_rect2)

    # Draw Quit button text
    button_text3 = font.render("X", True, white)
    text_rect3 = button_text3.get_rect(center=(30, 30))
    screen.blit(button_text3, text_rect3)


    pygame.time.Clock().tick(230)

        
