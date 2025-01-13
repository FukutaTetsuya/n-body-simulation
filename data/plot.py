import json
import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
l_max = 8
ax.set(aspect = "equal", xlim=[-l_max, +l_max], ylim=[-l_max, +l_max])
ims=[]

with open("data/history.txt", "r") as file:
    history = json.load(file)
    file.close()

for snapshot in history:
    x_list = snapshot["x"]
    y_list = snapshot["y"]
    im = plt.scatter(x_list, y_list, c="black", s=1)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=200)
ani.save("data/output.gif")
