from library import colorize, findsize
import cv2


img = cv2.imread("Test/5.jpg")




img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
# img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

# pix_pos, values = convert_and_extract_pixels(img)

clr = colorize(img)
clr = cv2.cvtColor(clr, cv2.COLOR_BGR2RGB)

cv2.imshow("original", img)
cv2.imshow("colorized", clr)


cv2.waitKey()
cv2.destroyAllWindows()


