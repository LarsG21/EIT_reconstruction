import time

import pygame



class KeyPressModule:
    def __init__(self):
        pygame.init()
        win = pygame.display.set_mode((400,400))

    def getKey(self, keyName):
        """
        Return true if a specific key is pressed currently
        :param keyName:
        :return:
        """
        ans = False
        for eve in pygame.event.get(): pass
        keyInput = pygame.key.get_pressed()
        myKey = getattr(pygame,'K_{}'.format(keyName))
        if keyInput[myKey]:
            ans = True
        pygame.display.update()

        return ans

    def get_keypress_down(self):
        """
        Return true if a specific key is pressed once
        :return:
        """
        ans = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                return event.dict["unicode"]


def main():
    kp = KeyPressModule()

    while True:
        key = kp.get_keypress_down()
        if (key is not None):
            print(key)
        # if kp.get_keypress_down('w'):
        #     print("Forward")
        # elif kp.get_keypress_down('s'):
        #     print("Backward")

if __name__ == "__main__":
    main()