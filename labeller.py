"""
Program for labelling resistors in a picture using the mouse
for input.
"""

import sys
from PointCollector import PointCollector
import matplotlib.pyplot as plt
import pylab

# Global point collector and pyplot
pc = None
fig = None
ax = None

# Mouse click callback
def onClick(event):
    global pc, fig, ax
    tb = plt.get_current_fig_manager().toolbar
    if tb.mode == "":
        if event.button == 1:
            pc.addPoint(event)
        elif event.button == 3:
            print("removing...")
            pc.removePoint(event)
        pc.draw()

def onKey(event):
    global pc
    if event.key == 'q':
        pc.save()
        sys.exit()
    elif event.key == 's':
        pc.save()


def main(argv):
    global pc, fig, ax

    if len(sys.argv) > 1:
        img_file = argv[1]
    else:
        print("Usage: python trainer.py INPUT_IMAGE")
        sys.exit()
    fig, ax = plt.subplots()
    pc = PointCollector(img_file, mode="rgb", fig=fig, ax=ax)
    plt.imshow(pc.img)
    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('key_press_event', onKey)
    pylab.show()

if __name__ == "__main__":
    main(sys.argv)
