from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def lakes_and_bays(image):
    b = ~image
    lb = label(b)
    regs = regionprops(lb)
    count_lakes = 0
    count_bays = 0
    for reg in regs:
        on_bound = False
        for y, x in reg.coords:
            if (y == 0 or x == 0 or y == image.shape[0] - 1
                    or x == image.shape[1] - 1):
                on_bound = True
                break
        if not on_bound:
            count_lakes += 1
        else:
            count_bays += 1
    return count_lakes, count_bays


def has_vline(region):
    lines = np.sum(region.image, 0) // region.image.shape[0]
    return 1 in lines


def filling_factor(region):
    return np.sum(region.image) / region.image.size


def recognize(region):
    if np.all(region.image):
        return "-"

    cl, cb = lakes_and_bays(region.image)

    if cl == 2:
        if (not has_vline(region)) or cb == 4:
            return "8"
        else:
            return "B"

    if cl == 1:
        if cb == 3:
            return "A"
        elif cb == 4:
            return "0"
        else:
            cut_cl, cut_cb = lakes_and_bays(region.image[0:14, 0:-1])
            if cut_cl > 0:
                return "P"
            return "D"

    if cl == 0:
        if has_vline(region):
            return "1"
        if cb == 2:
            return "/"
        cut_cl, cut_cb = lakes_and_bays(region.image[2:-2, 2:-2])
        if cut_cb == 4:
            return "X"
        if cut_cb == 5:
            cy = region.image.shape[0] // 2
            cx = region.image.shape[0] // 2
            if region.image[cy, cx] > 0:
                return "*"
            return "W"
    return None


t = perf_counter()

image = plt.imread("symbols.png")
binary = np.sum(image, 2)
binary[binary > 0] = 1

labeled = label(binary)

regions = regionprops(labeled)

# plt.imshow(regions[2].image,cmap="gray")

d = {}
for region in regions:
    symbol = recognize(region)
    if symbol is not None:
        labeled[np.where(labeled == region.label)] = 0
    else:
        print(filling_factor(region))
    if symbol not in d:
        d[symbol] = 0
    d[symbol] += 1

f = open("output.txt", "w")

f.write(f"Обработка заняла {perf_counter() - t} секунд\n")

if None not in d:
    f.write("Процент распознанного: 100%\n")
else:
    f.write(f"Процент распознанного: {round((1. - d[None] / sum(d.values())) * 100, 2)}\n")

sorted_tuples = sorted(d.items(), key=lambda item: item[1], reverse=True)
sorted_dict = {k: v for k, v in sorted_tuples}

f.write("Найденные объекты:\n")
for symbol in sorted_dict:
    f.write(f"Символ: {symbol}, количество: {d[symbol]}, "
            f"процент от всех найденных: {round(d[symbol] / sum(d.values()) * 100, 2)}\n")

f.close()

"""
debugging

object_to_see = i
print("Озёра и заливы на картинке:", lakes_and_bays(regions[object_to_see].image))
print("Процент заполнения на картинке:", filling_factor(regions[object_to_see]))
print("Распознается как:", recognize(regions[object_to_see]))
plt.imshow(regions[object_to_see].image)
plt.show()
"""
