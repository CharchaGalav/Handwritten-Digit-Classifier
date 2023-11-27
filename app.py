import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

windowSize = (500,500)

# Set up the colors 
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)

boundaryinc = 5
# Load the model
model = load_model('best_model')

labels = {
    0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6:'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'
}

imgsave = False
image_cnt = 1

PREDICT = True

pygame.init()

FONT = pygame.font.Font('freesansbold.ttf', 20)
displaysurface = pygame.display.set_mode(windowSize)

pygame.display.set_caption('Handwritten Digit Recognition')


# # Set up the background
# background = pygame.Surface(windowSize)
# background = background.convert()
# background.fill(WHITE)

iswriting = False

number_x = []
number_y = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            x, y = event.pos
            pygame.draw.circle(displaysurface, WHITE, (x,y), 4, 0)

            number_x.append(x)
            number_y.append(y)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_x = sorted(number_x)
            number_y = sorted(number_y)

            rect_min_x, rect_max_x = max(0, number_x[0] - boundaryinc), min(number_x[-1] + boundaryinc, windowSize[0])
            rect_min_y, rect_max_y = max(0, number_y[0] - boundaryinc), min(number_y[-1] + boundaryinc, windowSize[1])

            number_x = []
            number_y = []

            img_arr = np.array(pygame.PixelArray(displaysurface))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if imgsave:
                cv2.imwrite('handwrittenDigit/image.png')
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (12,12), 'constant', constant_values= 0)
                image = cv2.resize(image, (28,28))/255

                label = str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])

                textsurface =  FONT.render(label, True, RED, WHITE)
                textRectObj = textsurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_min_y
                displaysurface.blit(textsurface, textRectObj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    displaysurface.fill(BLACK)

        pygame.display.update()


